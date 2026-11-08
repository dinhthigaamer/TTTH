"""Validate instances and routes using the repository's exact mathematical model.

CLI: python scripts/validate_instances.py [instance.json | generated_dataset_dir]
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib
import json
import math
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/rescuenet_instances.json'
SOURCE_MANIFEST = ROOT/'configs/rescuenet_split.csv'
DEFAULT_DATA = ROOT/'data/instances/rescuenet_50x50'


def load_config():
    return json.loads(CONFIG.read_text(encoding='utf-8'))


def depot_for(image_id, seed):
    digest=hashlib.sha256(f'rescuenet-grid-v1:{image_id}:{seed}'.encode()).digest()
    rng=np.random.default_rng(int.from_bytes(digest[:8],'little'))
    return [int(rng.integers(50)),int(rng.integers(50))]


def depots_for(image_id, seed, K=6):
    """Uniform distinct cells, stable per image/seed, independent of risk/budget.

    Keep the historical reference cell as UAV 0 so B_base and D stay unchanged.
    Sample the remaining cells without replacement, excluding that first cell.
    """
    if not 1 <= K <= 2500:
        raise ValueError('Fleet size exceeds available depot cells')
    first=depot_for(image_id,seed)
    digest=hashlib.sha256(f'rescuenet-distinct-depots-v1:{image_id}:{seed}'.encode()).digest()
    rng=np.random.default_rng(int.from_bytes(digest[:8],'little'))
    remaining=np.delete(np.arange(2500),first[0]*50+first[1])
    sampled=rng.choice(remaining,size=K-1,replace=False)
    return [first]+[[int(cell//50),int(cell%50)] for cell in sampled]


def zigzag_reference(m,n,u,depot,craft):
    """Best orientation of a complete adjacent-cell serpentine sweep.

    Independent of risk/algorithm performance. Time ends at final arrival;
    energy includes all observations and the final return to the depot.
    """
    if m*n<2:
        raise ValueError('Budget reference needs at least two cells')
    endpoints=[]
    for rows,cols,transpose in [(m,n,False),(n,m,True)]:
        start=(0,0)
        end=(rows-1,cols-1 if rows%2 else 0)
        for flip_i in [False,True]:
            for flip_j in [False,True]:
                def transform(cell):
                    i,j=cell
                    i=rows-1-i if flip_i else i
                    j=cols-1-j if flip_j else j
                    return (j,i) if transpose else (i,j)
                endpoints.append((transform(start),transform(end)))
    internal=(m*n-1)*u
    choices=[]
    for start,end in endpoints:
        outbound=u*math.dist(depot,start)+internal
        back=u*math.dist(end,depot)
        energy=craft['e_h']*(outbound+back)+m*n*craft['eps']
        arrival=outbound/craft['v_h']+(m*n-1)*craft['tau']
        choices.append((energy,arrival,start,end,outbound,back))
    energy,arrival,start,end,outbound,back=min(choices)
    return {'energy':energy,'arrival_time':arrival,'outbound_distance':outbound,
            'return_distance':back,'start_cell':list(start),'last_cell':list(end),
            'observations':m*n,'method':'minimum-energy serpentine orientation'}


def budget_parameters(config,K,fraction,depot):
    """Return reference mean capacity B_base and common D from full-sweep costs."""
    grid=config['grid']
    ref=zigzag_reference(grid['m'],grid['n'],grid['u'],depot,config['uav'])
    B=round(fraction*ref['energy']/(K*config['eta']),6)
    D=round(fraction*ref['arrival_time']/K,6)
    return B,D,ref


def build_uavs(config,K,base_B,depots):
    """Distribute the old mean B without changing total fleet capacity.

    Round to integer micro-units first. Put rounding residual in the last UAV
    so the sum is exact at the six-decimal precision of the instance format.
    """
    if len(depots)!=K or any(len(cell)!=2 for cell in depots):
        raise ValueError('Expected one depot cell per UAV')
    factors=config['battery_allocation']['multipliers_by_K'][str(K)]
    if len(factors)!=K or any(not math.isfinite(x) or x<=0 for x in factors):
        raise ValueError('Invalid battery multipliers')
    decimals=[Decimal(str(x)) for x in factors]
    if sum(decimals)!=Decimal(K):
        raise ValueError('Battery multipliers must sum to K')
    base_micro=int((Decimal(str(base_B))*1000000).to_integral_value(rounding=ROUND_HALF_UP))
    values=[int((Decimal(base_micro)*factor).to_integral_value(rounding=ROUND_HALF_UP)) for factor in decimals]
    values[-1]+=K*base_micro-sum(values)
    if any(v<=0 for v in values):
        raise ValueError('Every UAV must have positive capacity')
    return [{'depot':cell.copy(),'B':v/1000000,**config['uav']} for v,cell in zip(values,depots)]


def number(value, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Expected finite numeric value (not bool/string)')
    if value < 0 or (positive and value == 0):
        raise ValueError('Expected positive/nonnegative value')
    return value


def validate_instance(instance):
    """Schema-equivalent field checks plus cross-field shape and feasibility."""
    keys = {'name','source','grid','K','D','eta','uavs','rho'}
    if not isinstance(instance, dict) or set(instance) != keys:
        raise ValueError('Missing or unexpected top-level fields')
    if not isinstance(instance['name'], str) or not instance['name'] or not isinstance(instance['source'], dict):
        raise ValueError('Invalid name/source')
    grid = instance['grid']
    if not isinstance(grid, dict) or set(grid) != {'m','n','u','w'}:
        raise ValueError('Invalid grid fields')
    m, n = grid['m'], grid['n']
    if any(type(x) is not int or x <= 0 for x in [m,n]):
        raise ValueError('Grid dimensions must be positive integers')
    u = number(grid['u'], positive=True)
    raw = grid['w']
    if (not isinstance(raw,list) or len(raw)!=m or any(not isinstance(r,list) or len(r)!=n for r in raw)
            or any(type(v) not in (int,float) for r in raw for v in r)):
        raise ValueError('w must be a numeric m x n array')
    w = np.asarray(raw, dtype=float)
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError('w must be finite and nonnegative')
    K = instance['K']
    if type(K) is not int or K <= 0 or not isinstance(instance['uavs'],list) or len(instance['uavs'])!=K:
        raise ValueError('K differs from UAV count')
    D = number(instance['D'], positive=True)
    eta = number(instance['eta'], positive=True)
    if eta > 1 or not isinstance(instance['rho'],dict):
        raise ValueError('Invalid eta/rho')
    if any(number(v)>1 for v in instance['rho'].values()):
        raise ValueError('rho must be in [0,1]')
    ii,jj = np.indices((m,n))
    for craft in instance['uavs']:
        if not isinstance(craft,dict) or set(craft)!={'depot','B','v_h','e_h','tau','eps'}:
            raise ValueError('Invalid UAV fields')
        depot = craft['depot']
        if (not isinstance(depot,list) or len(depot)!=2 or any(type(v) is not int for v in depot)
                or not 0<=depot[0]<m or not 0<=depot[1]<n):
            raise ValueError('Invalid depot')
        for key in ['B','v_h','e_h','tau','eps']:
            number(craft[key],positive=key in ['B','v_h','e_h'])
        distance = u*np.hypot(ii-depot[0],jj-depot[1])
        feasible = ((distance/craft['v_h']<=D+1e-9)&
                    (2*craft['e_h']*distance+craft['eps']<=eta*craft['B']+1e-9))
        # The format permits observing the depot. Do not add a restriction
        # that a feasible observation must be outside the depot.
        if not feasible.any():
            raise ValueError('UAV cannot observe a cell and return within budget')
    return w


def evaluate_routes(instance, routes):
    """Arrival deadline excludes final hover/return; energy includes return."""
    if not isinstance(routes,list) or len(routes)!=instance['K']:
        raise ValueError('Number of routes differs from K')
    m,n,u = (instance['grid'][k] for k in ['m','n','u'])
    covered,details = set(),[]
    for craft,route in zip(instance['uavs'],routes):
        if not isinstance(route,list):
            raise ValueError('Route must be a list')
        for cell in route:
            if (not isinstance(cell,list) or len(cell)!=2 or any(type(v) is not int for v in cell)
                    or not 0<=cell[0]<m or not 0<=cell[1]<n):
                raise ValueError('Invalid route cell')
        cells = [tuple(c) for c in route]
        if len(set(cells))!=len(cells):
            raise ValueError('Repeated cell in a route')
        if not cells:
            details.append({'cells':0,'arrival_time':0.,'energy':0.})
            continue
        previous = craft['depot']
        S = 0.
        for cell in cells:
            S += u*math.dist(previous,cell)
            previous = cell
        back = u*math.dist(previous,craft['depot'])
        arrival = S/craft['v_h']+(len(cells)-1)*craft['tau']
        energy = craft['e_h']*(S+back)+len(cells)*craft['eps']
        if arrival>instance['D']+1e-9 or energy>instance['eta']*craft['B']+1e-9:
            raise ValueError('Route exceeds deadline/energy')
        covered.update(cells)
        details.append({'cells':len(cells),'arrival_time':arrival,'energy':energy})
    total = float(np.asarray(instance['grid']['w']).sum())
    objective = float(sum(instance['grid']['w'][i][j] for i,j in covered))
    return {'objective':objective,'risk_coverage':objective/total if total else None,
            'cover_rate':len(covered)/(m*n),'uavs':details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path',nargs='?',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--solution',type=Path,help='Validate a solution for a single instance')
    args = parser.parse_args()
    schema = Draft202012Validator(json.loads((ROOT/'docs/instance.schema.json').read_text(encoding='utf-8')))
    if args.path.is_file():
        instance = json.loads(args.path.read_bytes())
        schema.validate(instance)
        validate_instance(instance)
        if args.solution:
            solution = json.loads(args.solution.read_bytes())
            if solution['instance']!=instance['name']:
                raise ValueError('Solution names a different instance')
            print(json.dumps(evaluate_routes(instance,solution['routes']),indent=2))
        else:
            print('Instance valid.')
        return
    base = args.path.resolve()
    config = load_config()
    source = pd.read_csv(SOURCE_MANIFEST,dtype={'id':str}).set_index('id')
    expected_config_hash = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    summary = json.loads((base/'summary.json').read_text(encoding='utf-8'))
    if summary['config_sha256']!=expected_config_hash:
        raise ValueError('Dataset was generated from a different configuration')
    frozen_source_hash=hashlib.sha256(SOURCE_MANIFEST.read_bytes()).hexdigest()
    if summary.get('frozen_split_manifest_sha256',frozen_source_hash)!=frozen_source_hash:
        raise ValueError('Frozen source split differs from the dataset provenance')
    expected_cases={(K,level['name'],seed) for K in config['K_values']
                    for level in config['budget_levels'] for seed in config['seeds']}
    expected_profiles={f'K{K}_b{level["name"]}' for K in config['K_values'] for level in config['budget_levels']}
    fractions={(K,level['name']):float(level['fractions_by_K'][str(K)])
               for K in config['K_values'] for level in config['budget_levels']}
    seen,per_image,by_split = set(),{},Counter()
    first_profiles = set()
    count = 0
    def check(row):
        instance = json.loads((base/row['path']).read_bytes())
        w = validate_instance(instance)
        src = instance['source']
        origin = source.loc[row['image_id']]
        combination=(row['K'],row['budget_level'],row['seed'])
        if combination not in expected_cases:
            raise ValueError('Unexpected K/budget/seed combination')
        fraction=fractions[(row['K'],row['budget_level'])]
        depots=depots_for(row['image_id'],row['seed'],row['K']);depot=depots[0]
        B,D,reference=budget_parameters(config,row['K'],fraction,depot)
        expected_name=f'rescuenet_{row["image_id"]}_50x50_K{row["K"]}_b{row["budget_level"]}_s{row["seed"]}'
        saved_reference=src['budget_reference']
        reference_matches=set(saved_reference)==set(reference) and all(
            math.isclose(saved_reference[k],value,rel_tol=0,abs_tol=1e-8)
            if type(value) in (int,float) else saved_reference[k]==value
            for k,value in reference.items())
        if (instance['name']!=row['name'] or src['image_id']!=row['image_id']
                or src['split']!=row['split'] or origin['split']!=row['split']
                or src['group_id']!=origin.group_id or src['seed']!=row['seed']
                or src['mask_sha256']!=origin.mask_sha256
                or src['image_path']!=origin.image_path or src['mask_path']!=origin.mask_path
                or instance['rho']!=config['rho'] or instance['eta']!=config['eta'] or instance['grid']['u']!=40
                or w.shape!=(50,50) or instance['K']!=row['K'] or abs(instance['D']-D)>1e-8
                or abs(row['D']-D)>1e-8 or abs(row['B']-B)>1e-8
                or row['profile']!=f'K{row["K"]}_b{row["budget_level"]}' or row['name']!=expected_name
                or src['budget_level']!=row['budget_level'] or src['budget_fraction']!=fraction
                or abs(row['budget_fraction']-fraction)>1e-12 or not reference_matches
                or src['budget_rule']!=config['budget_rule'] or src['config_version']!=config['version']
                or [row['reference_depot_i'],row['reference_depot_j']]!=depot
                or json.loads(row['depot_values'])!=depots
                or src['depot_policy']!=config['depot_allocation']
                or Path(row['path']).stem!=expected_name):
            raise ValueError(f'Configuration/source mismatch: {row["path"]}')
        expected_uavs=build_uavs(config,row['K'],B,depots)
        if instance['uavs']!=expected_uavs:
            raise ValueError('Unexpected per-UAV parameters/battery allocation')
        capacities=[u['B'] for u in expected_uavs]
        if (abs(row['B_min']-min(capacities))>1e-8 or abs(row['B_max']-max(capacities))>1e-8
                or abs(row['B_team']-row['K']*B)>1e-6
                or json.loads(row['B_values'])!=capacities
                or src['battery_multipliers']!=config['battery_allocation']['multipliers_by_K'][str(row['K'])]):
            raise ValueError('Incorrect heterogeneous battery metadata')
        actual = hashlib.sha256(np.round(w*72000).astype('<u4').tobytes()).hexdigest()
        if actual!=row['grid_sha256'] or src['grid_sha256']!=actual:
            raise ValueError('Grid does not match manifest fingerprint')
        if (abs(float(w.sum())-row['total_risk'])>1e-8
                or src['zero_risk']!=(w.sum()==0)):
            raise ValueError('Incorrect risk statistics')
        return row['image_id'],row['split'],combination
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for chunk in pd.read_csv(base/'manifest.csv',dtype={'image_id':str},chunksize=1200):
            if chunk.name.duplicated().any() or seen.intersection(chunk.name):
                raise ValueError('Duplicate instance names')
            seen.update(chunk.name)
            if len(first_profiles)<len(expected_profiles):
                for _,row in chunk.iterrows():
                    if row['profile'] not in first_profiles:
                        schema.validate(json.loads((base/row['path']).read_bytes()))
                        first_profiles.add(row['profile'])
            for image_id,split,combination in pool.map(check,chunk.to_dict('records')):
                combinations=per_image.setdefault(image_id,set())
                if combination in combinations:
                    raise ValueError('Duplicate parameter combination for one image')
                combinations.add(combination);by_split[split]+=1;count+=1
            if count%12000==0:
                print(f'Checked {count:,}/{summary["instances"]:,}',flush=True)
    if (count!=len(source)*len(expected_cases) or set(per_image)!=set(source.index)
            or any(cases!=expected_cases for cases in per_image.values())
            or any(source.groupby('group_id')['split'].nunique()>1)):
        raise ValueError('Incomplete dataset or source-group overlap')
    actual_files = sum(1 for split in ['val','test'] for _ in (base/split).rglob('*.json'))
    if actual_files!=count:
        raise ValueError('Extra/missing JSON files')
    summary['validation']={'all_checks_passed':True,'json_files_checked':count,
                           'schema_profiles_checked':len(first_profiles),'by_split':dict(by_split),
                           'budget_rule_recomputed_for_every_file':True,
                           'instances_per_image':len(expected_cases),
                           'rules':'docs/instance_format.md; deadline to last arrival; return-inclusive energy'}
    summary['frozen_split_manifest_file']='configs/rescuenet_split.csv'
    summary['frozen_split_manifest_sha256']=frozen_source_hash
    (base/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary['validation'],indent=2),flush=True)


if __name__=='__main__':
    main()
