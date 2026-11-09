"""Check native-pixel aggregation and the model's easy-to-confuse constraints."""
import copy
import json
from pathlib import Path
import sys
import unittest
import itertools
import tempfile
import hashlib

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from generate_rescuenet_instances import native_to_grid, weights_from_numerators, read_source, encode_instance, create_image
from validate_instances import evaluate_routes, validate_instance, zigzag_reference, budget_parameters, load_config, build_uavs, depots_for, config_path_for, depot_for
from evaluate_instances import group_summary
from coverage_certificate import CoverageBound,route_ticks,sweep_routes,impossibility_certificate


def toy():
    return {'name': 'boundary', 'source': {}, 'grid': {'m': 2, 'n': 2, 'u': 40,
            'w': [[0, .25], [.5, 1.]]}, 'K': 1, 'D': 40/15, 'eta': 1.,
            'uavs': [{'depot': [0, 0], 'B': 80, 'v_h': 15, 'e_h': 1, 'tau': 100, 'eps': 0}],
            'rho': {'a': 0., 'b': 1.}}


class NativeGridTests(unittest.TestCase):
    def test_unselected_source_generates_no_instance_json(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);mask=np.zeros((3000,3000),dtype=np.uint8)
            path=root/'zero.png';Image.fromarray(mask).save(path)
            row={'id':'synthetic_zero','split':'val','original_split':'train','group_id':'synthetic',
                 'image_path':'unused.jpg','mask_path':str(path),'mask_sha256':hashlib.sha256(mask.tobytes()).hexdigest()}
            rows,zero,proofs,rejected=create_image((row,load_config(),str(root/'instances')))
            self.assertTrue(zero)
            self.assertEqual((rows,proofs),([],[]))
            self.assertEqual(rejected['status'],'proven_incompatible')
            self.assertEqual(list((root/'instances').rglob('*.json')),[])

    def test_selected_image_writes_only_nine_certified_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);mask=np.full((3000,3000),5,dtype=np.uint8)
            path=root/'positive.png';Image.fromarray(mask).save(path)
            base=root/'instances'
            for level in ['low','medium','high']:(base/'val'/f'K6_b{level}').mkdir(parents=True)
            row={'id':'synthetic_uniform','split':'val','original_split':'train','group_id':'synthetic',
                 'image_path':'unused.jpg','mask_path':str(path),'mask_sha256':hashlib.sha256(mask.tobytes()).hexdigest()}
            rows,zero,proofs,rejected=create_image((row,load_config(),str(base)))
            self.assertIsNone(rejected)
            self.assertFalse(zero)
            self.assertEqual(len(rows),9)
            self.assertEqual(len(list(base.rglob('*.json'))),9)
            self.assertFalse((base/'edge_cases').exists())
            for proof in proofs:
                self.assertGreaterEqual(proof['lower_ticks']*10,proof['total_ticks'])
                self.assertLessEqual(proof['upper_ticks']*2,proof['total_ticks'])

    def test_100_grid_preserves_crop_and_exactly_refines_50_grid(self):
        mask=np.zeros((3040,4056),dtype=np.uint8)
        # Adjacent 30-pixel cells have distinct labels; outer pixels are excluded.
        mask[:20,:]=5
        mask[20:50,528:558]=5
        mask[20:50,558:588]=3
        mask[50:80,528:558]=9
        mask[50:80,558:588]=8
        scale=np.array([0,2,0,5,15,20,0,0,4,4,0],dtype=np.uint8)
        a,crop50=native_to_grid(mask,scale,50)
        b,crop100=native_to_grid(mask,scale,100)
        self.assertEqual(crop50,crop100)
        self.assertEqual(b.shape,(100,100))
        np.testing.assert_array_equal(b.reshape(50,2,50,2).sum(axis=(1,3)),a)
        w=weights_from_numerators(b)
        np.testing.assert_array_equal(w[:2,:2],[[1,.25],[.2,.2]])
        np.testing.assert_allclose(w.reshape(50,2,50,2).mean(axis=(1,3)),weights_from_numerators(a),atol=1e-12,rtol=0)

    def test_100_depots_and_budgets_use_actual_grid(self):
        c=load_config(config_path_for(100))
        self.assertEqual(c['grid'],{'m':100,'n':100,'u':40})
        cells=depots_for('10778',42,6,100,100)
        self.assertEqual(cells,depots_for('10778',42,6,100,100))
        self.assertEqual(len(set(map(tuple,cells))),6)
        self.assertTrue(all(0<=x<100 for cell in cells for x in cell))
        self.assertTrue(any(x>=50 for cell in cells for x in cell))
        B,D,ref=budget_parameters(c,6,.125,cells[0])
        self.assertEqual(ref['observations'],10000)
        self.assertAlmostEqual(6*B,.125*ref['energy'],delta=4e-6)
        self.assertAlmostEqual(6*D,.125*ref['arrival_time'],delta=4e-6)
        fleet=build_uavs(c,6,B,cells)
        self.assertEqual(sum(round(u['B']*1e6) for u in fleet),6*round(B*1e6))

    def test_example_50_depot_and_budget_match_config(self):
        example=Path(__file__).resolve().parents[1]/'docs/examples/rescuenet_10778_50x50_K6_blow_s42.json'
        x=json.loads(example.read_text(encoding='utf-8'));c=load_config()
        cells=depots_for('10778',42)
        fraction=x['source']['budget_fraction']
        B,D,_=budget_parameters(c,6,fraction,cells[0],x['source'].get('budget_time_fraction',fraction))
        self.assertEqual(x['uavs'],build_uavs(c,6,B,cells))
        self.assertEqual(x['D'],D)

    def test_orientation_crop_and_mixed_pixel_mean(self):
        mask = np.zeros((3040, 4056), dtype=np.uint8)
        # center crop starts at x=528,y=20. Distinguish rows from columns.
        mask[20:80, 528:588] = 5
        mask[20:80, 588:618] = 3  # half of cell [0,1] is minor damage
        mask[80:140, 528:588] = 1
        scale = np.array([0, 2, 0, 5, 15, 20, 0, 0, 4, 0, 0], dtype=np.uint8)
        numerators, crop = native_to_grid(mask, scale)
        w = weights_from_numerators(numerators)
        self.assertEqual(crop, [528, 20, 3000, 3000])
        self.assertEqual(w.shape, (50, 50))
        self.assertEqual(w[0, 0], 1.)
        self.assertEqual(w[0, 1], .125)
        self.assertEqual(w[1, 0], .1)
        self.assertEqual(w[1, 1], 0.)
        self.assertAlmostEqual(w.sum(), 1.225)

    def test_invalid_label_is_not_silently_mapped(self):
        mask = np.zeros((3000, 3000), dtype=np.uint8)
        mask[0, 0] = 255
        with self.assertRaises(ValueError):
            native_to_grid(mask, np.zeros(11, dtype=np.uint8))

    def test_depot_is_stable(self):
        self.assertEqual(depot_for('10778', 42), depot_for('10778', 42))
        self.assertTrue(all(0 <= x < 50 for x in depot_for('10778', 42)))

    def test_distinct_depots_reproducible_and_independent_of_budget(self):
        cells=depots_for('10778',42)
        self.assertEqual(cells,depots_for('10778',42))
        self.assertEqual(len(set(map(tuple,cells))),6)
        self.assertEqual(cells[0],depot_for('10778',42))
        self.assertNotEqual(cells,depots_for('10778',43))
        self.assertNotEqual(cells,depots_for('10779',42))
        self.assertTrue(all(0<=x<50 for cell in cells for x in cell))
        c=load_config()
        fleets=[build_uavs(c,6,b,cells) for b in [1000,2000,3000]]
        self.assertTrue(all([u['depot'] for u in fleet]==cells for fleet in fleets))

    def test_routes_return_to_each_uavs_own_depot(self):
        instance=toy();instance['K']=2;instance['D']=1000
        instance['uavs'].append({**copy.deepcopy(instance['uavs'][0]),'depot':[1,1]})
        result=evaluate_routes(instance,[[[0,1]],[[1,0]]])
        self.assertEqual([r['energy'] for r in result['uavs']],[80,80])

    def test_frozen_split_is_complete_and_both_grids_require_certification(self):
        source=read_source()
        self.assertEqual(len(source),4494)
        self.assertEqual(source.split.value_counts().to_dict(),{'test':3595,'val':899})
        self.assertTrue(source.groupby('group_id').split.nunique().eq(1).all())
        self.assertTrue(source.merged_mask_path.equals(source.mask_path))
        for size in (50,100):
            self.assertTrue(load_config(config_path_for(size))['selection']['only_certified'])

    def test_tree_update_changes_only_tree_pixels(self):
        mask = np.zeros((3000, 3000), dtype=np.uint8)
        mask[:60, :60] = 9
        mask[:60, 60:90] = 9
        mask[:60, 90:120] = 3
        mask[60:120, :60] = 8
        old_scale = np.array([0, 2, 0, 5, 15, 20, 0, 0, 4, 0, 0], dtype=np.uint8)
        new_scale = old_scale.copy()
        new_scale[9] = 4
        old_w = weights_from_numerators(native_to_grid(mask, old_scale)[0])
        new_w = weights_from_numerators(native_to_grid(mask, new_scale)[0])
        self.assertEqual(new_w[0, 0], .2)
        self.assertEqual(new_w[0, 1], .225)  # half Tree .2, half Minor .25
        self.assertEqual(new_w[1, 0], old_w[1, 0])  # Road-Blocked stays .2
        expected_delta = np.zeros((50, 50))
        expected_delta[0, :2] = [.2, .1]
        np.testing.assert_allclose(new_w-old_w, expected_delta, atol=1e-12, rtol=0)


class ModelTests(unittest.TestCase):
    def test_certificate_upper_bound_dominates_exact_small_instance_optimum(self):
        cells=[[i,j] for i in range(2) for j in range(3)]
        for B,D in [(85,5),(170,15),(320,30)]:
            x={'name':'exact','source':{},'grid':{'m':2,'n':3,'u':40,'w':[[.1,.8,.2],[.3,0.,.6]]},
               'K':2,'D':D,'eta':1,'rho':{},'uavs':[
                   {'depot':[0,0],'B':B,'v_h':15,'e_h':1,'tau':2,'eps':0},
                   {'depot':[1,2],'B':B*.8,'v_h':15,'e_h':1,'tau':2,'eps':0}]}
            feasible=[]
            for k in range(2):
                routes=[]
                for length in range(7):
                    for path in itertools.permutations(cells,length):
                        route=[list(c) for c in path]
                        candidates=[[],[]];candidates[k]=route
                        try:evaluate_routes(x,candidates)
                        except ValueError:continue
                        routes.append(route)
                feasible.append(routes)
            optimum=max(route_ticks(x,[a,b]) for a in feasible[0] for b in feasible[1])
            for method in ['reachable_top_weights_v1','entry_distance_top_weights_v2']:
                upper=CoverageBound(np.asarray(x['grid']['w']),40,[[0,0],[1,2]],method).upper(x)
                self.assertGreaterEqual(upper['upper_ticks'],optimum)

    def test_entry_bound_covers_exact_optimum_for_remote_sparse_cells(self):
        cells=[[3,1],[3,4]];w=np.zeros((4,5));w[3,1]=.4;w[3,4]=.6
        for B,D in [(200,10),(300,20),(500,40)]:
            x={'name':'remote','source':{},'grid':{'m':4,'n':5,'u':40,'w':w.tolist()},
                'K':2,'D':D,'eta':1,'rho':{},'uavs':[
                    {'depot':[0,0],'B':B,'v_h':15,'e_h':1,'tau':2,'eps':0},
                    {'depot':[0,4],'B':B,'v_h':15,'e_h':1,'tau':2,'eps':0}]}
            paths=[list(p) for L in range(3) for p in itertools.permutations(cells,L)]
            optimum=0
            for a in paths:
                for b in paths:
                    try:evaluate_routes(x,[a,b])
                    except ValueError:continue
                    optimum=max(optimum,route_ticks(x,[a,b]))
            old=CoverageBound(w,40,[[0,0],[0,4]]).upper(x)
            new=CoverageBound(w,40,[[0,0],[0,4]],'entry_distance_top_weights_v2').upper(x)
            self.assertGreaterEqual(new['upper_ticks'],optimum)
            self.assertLessEqual(new['upper_ticks'],old['upper_ticks'])

    def test_dominant_cell_impossibility_is_not_a_search_failure(self):
        c=load_config(config_path_for(100));w=np.zeros((100,100));w[0,1]=.6;w[0,2]=.4
        depots=[[k,0] for k in range(6)]
        p=impossibility_certificate(c,w,depots)
        self.assertEqual(p['reason'],'dominant_cell_forced_by_minor_reachability')
        self.assertEqual(len(p['blocker_uavs']),6)
        w[0,1],w[0,2]=.4,.6
        self.assertIsNone(impossibility_certificate(c,w,depots))

    def test_time_and_energy_fractions_can_be_selected_independently(self):
        c=load_config(config_path_for(100))
        B,D,ref=budget_parameters(c,6,.04,[10,20],.1)
        old_B,_,_=budget_parameters(c,6,.04,[10,20])
        self.assertEqual(B,old_B)
        self.assertEqual(D,round(.1*ref['arrival_time']/6,6))

    def test_certificate_includes_free_depot_observations(self):
        x=toy();x['D']=.001;x['uavs'][0]['B']=.001
        x['grid']['w']=[[.9,0],[0,.1]]
        upper=CoverageBound(np.asarray(x['grid']['w']),40,[[0,0]]).upper(x)
        self.assertEqual(upper['upper_ticks'],900000000000)

    def test_sweep_witness_respects_each_depot_and_battery(self):
        x=toy();x['D']=20;x['uavs'][0]['B']=200
        routes=sweep_routes(x)
        evaluate_routes(x,routes)
        self.assertGreater(route_ticks(x,routes),0)

    def test_mild_heterogeneous_capacities_preserve_exact_team_budget(self):
        c=load_config();base=2135.21385
        fleet=build_uavs(c,6,base,depots_for("10778",42))
        self.assertEqual(len({u['B'] for u in fleet}),6)
        self.assertEqual(sum(round(u['B']*1e6) for u in fleet),6*round(base*1e6))
        for craft,factor in zip(fleet,[.8,.9,.95,1.05,1.1,1.2]):
            self.assertAlmostEqual(craft['B'],base*factor,delta=2e-6)
            self.assertIn(craft['depot'],depots_for('10778',42))
            for key,value in c['uav'].items():self.assertEqual(craft[key],value)

    def test_uavs_have_one_readable_object_per_line(self):
        instance=toy();instance['K']=6
        instance['uavs']=build_uavs(load_config(),6,3000,depots_for('10778',42))
        fields={k:v for k,v in instance.items() if k!='grid'}
        text=encode_instance(fields,json.dumps(instance['grid'])).decode('utf-8')
        self.assertEqual(json.loads(text),instance)
        lines=text.split('"uavs": [\n',1)[1].split('\n  ]',1)[0].splitlines()
        self.assertEqual(len(lines),6)
        self.assertEqual([json.loads(line.strip().rstrip(',')) for line in lines],instance['uavs'])

    def test_each_uav_uses_its_own_battery_limit(self):
        instance=toy();instance['D']=1000;instance['K']=2
        instance['uavs'].append({**copy.deepcopy(instance['uavs'][0]),'B':120})
        evaluate_routes(instance,[[],[[1,1]]])
        with self.assertRaises(ValueError):evaluate_routes(instance,[[[1,1]],[]])

    def test_coverage_summary_excludes_zero_risk_and_averages_seeds_per_image(self):
        records=[]
        for image_id,total,ratios in [('a',10.,[.1,.3,.5]),('b',20.,[.8,.8,.8]),('zero',0.,[None]*3)]:
            for ratio in ratios:
                records.append({'K':np.int64(1),'budget_level':'low','image_id':image_id,
                    'total_risk':total,'covered_risk':total*ratio if ratio is not None else 0.,'risk_coverage':ratio})
        result=group_summary(pd.DataFrame(records),['K','budget_level'])[0]
        self.assertAlmostEqual(result['mean_coverage_pct'],55.)
        self.assertEqual(result['zero_risk_instances'],3)
        self.assertEqual(result['positive_images'],2)
        json.dumps(result)  # NumPy group keys must also be serializable.

    def test_zigzag_reference_matches_complete_route(self):
        craft={'depot':[0,0],'B':10000,'v_h':15,'e_h':2,'tau':3,'eps':5}
        instance={'grid':{'m':2,'n':3,'u':40,'w':[[1,1,1],[1,1,1]]},
                  'K':1,'D':1000,'eta':1.,'uavs':[craft]}
        result=evaluate_routes(instance,[[[0,0],[0,1],[0,2],[1,2],[1,1],[1,0]]])
        reference=zigzag_reference(2,3,40,[0,0],craft)
        self.assertAlmostEqual(reference['energy'],510.)
        self.assertAlmostEqual(reference['energy'],result['uavs'][0]['energy'])
        self.assertAlmostEqual(reference['arrival_time'],result['uavs'][0]['arrival_time'])
        self.assertEqual(reference['observations'],6)

    def test_budget_is_fraction_of_zigzag_including_safety_factor(self):
        c={'grid':{'m':2,'n':3,'u':40},'eta':.9,
           'uav':{'v_h':15,'e_h':2,'tau':3,'eps':5}}
        B,D,ref=budget_parameters(c,6,.25,[0,0])
        self.assertAlmostEqual(.9*B*6,.25*ref['energy'],delta=5e-6)
        self.assertAlmostEqual(D*6,.25*ref['arrival_time'],delta=5e-6)

    def test_agreed_input_configuration(self):
        c=load_config()
        self.assertEqual(c['seeds'],[42,43,44])
        self.assertEqual(c['K_values'],[6])
        self.assertEqual(c['battery_allocation']['multipliers_by_K'],{'6':[.8,.9,.95,1.05,1.1,1.2]})
        self.assertEqual(c['rho'],{'Background':0.,'Water':.1,'Building No Damage':0.,
            'Building Minor Damage':.25,'Building Major Damage':.75,'Building Total Destruction':1.,
            'Vehicle':0.,'Road-Clear':0.,'Road-Blocked':.2,'Tree':.2,'Pool':0.})
        self.assertEqual([x['name'] for x in c['budget_levels']],['low','medium','high'])
        self.assertTrue(c['selection']['only_certified'])
        self.assertLess(c['coverage_certification']['alpha_min'],c['coverage_certification']['alpha_max'])

    def test_last_arrival_excludes_last_hover_but_return_costs_energy(self):
        instance = toy()
        validate_instance(instance)
        result = evaluate_routes(instance, [[[0, 1]]])
        self.assertAlmostEqual(result['uavs'][0]['arrival_time'], 40/15)
        self.assertEqual(result['uavs'][0]['energy'], 80)
        self.assertEqual(result['objective'], .25)
        instance['uavs'][0]['B'] = 79.9
        with self.assertRaises(ValueError):
            evaluate_routes(instance, [[[0, 1]]])

    def test_safety_factor_and_diagonal_distance(self):
        instance = toy()
        instance['D'] = 1000
        instance['uavs'][0]['B'] = 80*np.sqrt(2)
        self.assertAlmostEqual(evaluate_routes(instance, [[[1, 1]]])['uavs'][0]['energy'], 80*np.sqrt(2))
        instance['eta'] = .9
        with self.assertRaises(ValueError):
            evaluate_routes(instance, [[[1, 1]]])

    def test_shared_cells_are_rewarded_once_and_empty_routes_are_valid(self):
        instance = toy()
        instance['K'] = 2
        instance['uavs'].append(copy.deepcopy(instance['uavs'][0]))
        self.assertEqual(evaluate_routes(instance, [[[0, 1]], [[0, 1]]])['objective'], .25)
        self.assertEqual(evaluate_routes(instance, [[], []])['objective'], 0.)
        instance['grid']['w'] = [[0, 0], [0, 0]]
        self.assertIsNone(evaluate_routes(instance, [[], []])['risk_coverage'])

    def test_duplicate_and_out_of_grid_cells_rejected(self):
        for route in [[[0, 1], [0, 1]], [[2, 0]]]:
            with self.assertRaises(ValueError):
                evaluate_routes(toy(), [route])

    def test_cross_field_validation(self):
        instance = toy()
        instance['K'] = 2
        with self.assertRaises(ValueError):
            validate_instance(instance)

    def test_depot_observation_is_allowed_by_format(self):
        instance = toy()
        instance['grid'] = {'m':1,'n':1,'u':40,'w':[[1.]]}
        instance['uavs'][0]['B'] = .01
        validate_instance(instance)
        self.assertEqual(evaluate_routes(instance,[[[0,0]]])['objective'],1.)

    def test_nonnumeric_grid_values_are_rejected(self):
        for value in [True,'0.2',float('nan'),float('inf'),-.1]:
            instance = toy()
            instance['grid']['w'][0][0] = value
            with self.assertRaises(ValueError):
                validate_instance(instance)


if __name__ == '__main__':
    unittest.main()
