"""Check native-pixel aggregation and the model's easy-to-confuse constraints."""
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from generate_rescuenet_instances import native_to_grid, weights_from_numerators, depot_for, read_source, encode_instance
from validate_instances import evaluate_routes, validate_instance, zigzag_reference, budget_parameters, load_config, build_uavs, depots_for
from evaluate_instances import group_summary


def toy():
    return {'name': 'boundary', 'source': {}, 'grid': {'m': 2, 'n': 2, 'u': 40,
            'w': [[0, .25], [.5, 1.]]}, 'K': 1, 'D': 40/15, 'eta': 1.,
            'uavs': [{'depot': [0, 0], 'B': 80, 'v_h': 15, 'e_h': 1, 'tau': 100, 'eps': 0}],
            'rho': {'a': 0., 'b': 1.}}


class NativeGridTests(unittest.TestCase):
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

    def test_committed_split_is_complete_and_calibration_is_val_only(self):
        source=read_source()
        self.assertEqual(len(source),4494)
        self.assertEqual(source.split.value_counts().to_dict(),{'test':3595,'val':899})
        self.assertTrue(source.groupby('group_id').split.nunique().eq(1).all())
        self.assertTrue(source.merged_mask_path.equals(source.mask_path))
        calibration_ids=load_config()['calibration']['image_ids']
        self.assertTrue(source.set_index('id').loc[calibration_ids,'split'].eq('val').all())

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
        for K in ['6']:
            fractions=[x['fractions_by_K'][K] for x in c['budget_levels']]
            self.assertTrue(0<fractions[0]<fractions[1]<fractions[2]<=1)

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
