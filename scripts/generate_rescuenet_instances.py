"""Generate RescueNet inputs using zig-zag-normalized energy/time budgets.

One image -> one 50x50 risk map -> K=6 x 3 budget levels x 3 depot seeds.
Configuration is in configs/rescuenet_instances.json. Source splits stay frozen.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image
from jsonschema import Draft202012Validator

from validate_instances import (ROOT,CONFIG,SOURCE_MANIFEST,DEFAULT_DATA,load_config,validate_instance,
                                evaluate_routes,depot_for,depots_for,budget_parameters,build_uavs)

M=N=50
CELL=60
SQUARE=3000
DENOMINATOR=72000


def native_to_grid(mask,rho_scaled):
    a=np.asarray(mask)
    if a.ndim!=2 or a.dtype.kind not in 'iu' or a.min()<0 or a.max()>10:
        raise ValueError('Expected integer mask IDs 0..10')
    h,w=a.shape
    if min(h,w)<SQUARE:
        raise ValueError('Mask smaller than 3000x3000')
    x,y=(w-SQUARE)//2,(h-SQUARE)//2
    crop=a[y:y+SQUARE,x:x+SQUARE]
    numerator=rho_scaled[crop].reshape(M,CELL,N,CELL).sum(axis=(1,3),dtype=np.uint32)
    return numerator,[x,y,SQUARE,SQUARE]


def weights_from_numerators(numerators):
    return np.round(numerators.astype(float)/DENOMINATOR,12)


def weight_vectors(config):
    values=np.array([config['rho'][name] for name in config['class_names']],dtype=float)
    scaled=np.rint(values*20).astype(np.uint8)
    if (len(values)!=11 or not np.isfinite(values).all() or (values<0).any() or (values>1).any()
            or not np.allclose(scaled/20,values,atol=1e-15,rtol=0)):
        raise ValueError('Weights must be in [0,1] and exactly representable by scale 20')
    return values,scaled


def read_source():
    source=pd.read_csv(SOURCE_MANIFEST,dtype={'id':str}).sort_values('id').reset_index(drop=True)
    # The committed manifest refers directly to original labels. No hardlinked
    # data/RescueNet_merged directory or local EDA outputs are required to generate.
    source['merged_mask_path']=source['mask_path']
    if (len(source)!=4494 or not source.id.is_unique or set(source.split)!={'val','test'}
            or (source.groupby('group_id').split.nunique()>1).any()
            or source.split.value_counts().to_dict()!={'test':3595,'val':899}):
        raise ValueError('Frozen source split is missing or inconsistent')
    return source


def greedy_routes(instance):
    """Deterministic greedy for calibration/evaluation; not an optimum."""
    w=np.asarray(instance['grid']['w']);cells=np.column_stack(np.nonzero(w>0))
    values=w[cells[:,0],cells[:,1]] if len(cells) else np.array([])
    available=np.ones(len(cells),dtype=bool);routes=[]
    for craft in instance['uavs']:
        cur=np.array(craft['depot']);route=[];flown=arrival=0.
        back=40*np.linalg.norm(cells-cur,axis=1)
        while available.any():
            distance=40*np.linalg.norm(cells-cur,axis=1)
            t=arrival+distance/craft['v_h']+(craft['tau'] if route else 0)
            e=craft['e_h']*(flown+distance+back)+(len(route)+1)*craft['eps']
            feasible=available&(t<=instance['D']+1e-9)&(e<=instance['eta']*craft['B']+1e-9)
            if not feasible.any():
                break
            best=int(np.argmax(np.where(feasible,values/(distance/40+1),-np.inf)))
            flown+=float(distance[best]);arrival=float(t[best]);cur=cells[best]
            route.append(cur.tolist());available[best]=False
        routes.append(route)
    return routes


def calibrate_budgets(config,source,max_maps=32):
    """Optional new calibration using val only, all 3 distinct-depot seeds."""
    _,scaled=weight_vectors(config)
    selected=[]
    val=source[source.split=='val'].sample(frac=1,random_state=42)
    for group_id,group in val.groupby('group_id',sort=False):
        for row in group.to_dict('records'):
            with Image.open(ROOT/row['merged_mask_path']) as im:
                w=weights_from_numerators(native_to_grid(im,scaled)[0])
            if w.sum()>0:
                selected.append((row['id'],group_id,w));break
        if len(selected)==max_maps:
            break
    if len(selected)<12:
        raise ValueError('Too few positive val groups for calibration')
    candidates={6:[.075,.09,.1,.11,.125,.15,.175,.2,.225,.24,.25,.275]}
    samples=[];chosen_by_K={};selected_stats=[]
    for K in config['K_values']:
        for fraction in candidates[K]:
            for seed in config['seeds']:
                for image_id,group_id,w in selected:
                    depot=depot_for(image_id,seed)
                    B,D,_=budget_parameters(config,K,fraction,depot)
                    instance={'grid':{**config['grid'],'w':w.tolist()},'K':K,'D':D,'eta':config['eta'],
                              'uavs':build_uavs(config,K,B,depots_for(image_id,seed,K))}
                    result=evaluate_routes(instance,greedy_routes(instance))
                    samples.append({'K':K,'fraction':fraction,'seed':seed,'coverage':result['risk_coverage']})
            recent=pd.DataFrame(samples).query('K==@K and fraction==@fraction')
            print(f'Val K={K}, fraction={fraction}: '+str(recent.groupby('seed').coverage.median().round(4).to_dict()),flush=True)
        data=pd.DataFrame(samples).query('K==@K')
        med=data.groupby(['fraction','seed']).coverage.median().unstack()
        eligible=[float(f) for f in med.index if med.loc[f].between(.1,.5).all()]
        if len(eligible)<3:
            raise ValueError(f'Need at least 3 fractions with 10-50% median for all seeds at K={K}; got {eligible}')
        targets=[.15,.30,.45]
        options=[]
        for triple in itertools.combinations(eligible,3):
            loss=sum(float(((med.loc[f].to_numpy()-target)**2).sum()) for f,target in zip(triple,targets))
            options.append((loss,triple))
        chosen_by_K[K]=list(min(options)[1])
        for label,fraction in zip(['low','medium','high'],chosen_by_K[K]):
            sub=data[data.fraction==fraction]
            selected_stats.append({'K':K,'level':label,'fraction':fraction,
                                   'median_risk_coverage':float(sub.coverage.median()),
                                   'median_by_seed':{str(seed):float(med.loc[fraction,seed]) for seed in config['seeds']},
                                   'share_cases_in_10_to_50pct':float(sub.coverage.between(.1,.5).mean())})
    config['budget_levels']=[{'name':name,'fractions_by_K':{str(K):chosen_by_K[K][i] for K in config['K_values']}}
                             for i,name in enumerate(['low','medium','high'])]
    config['calibration']={'split':'val_only','positive_maps':len(selected),
        'image_ids':[i for i,g,w in selected],'group_ids':[g for i,g,w in selected],
        'seeds':config['seeds'],'candidate_fractions_by_K':{str(k):v for k,v in candidates.items()},
        'cases_checked':len(samples),'selected_levels':selected_stats,
        'method':'greedy gain/(distance_in_cells+1); every route independently checked',
        'allocation_during_search':config['battery_allocation'],
        'depot_allocation_during_search':config['depot_allocation'],
        'selection':'eligible if per-seed median is 10-50%; minimize squared differences to 15/30/45% targets',
        'interpretation':'K6 distinct-depot val calibration; test never used for parameter selection',
        'limitation':'heuristic val pilot; not an optimum and not a 10-50% guarantee for each image'}
    print('Chosen budget levels: '+json.dumps(config['budget_levels']),flush=True)
    return config


def write_identical_or_new(path,payload):
    if path.exists():
        if path.read_bytes()!=payload:
            raise FileExistsError(f'Existing file differs; use a clean destination: {path}')
        return
    temporary=path.with_suffix(path.suffix+'.part')
    temporary.write_bytes(payload);os.replace(temporary,path)


def encode_instance(data,encoded_grid):
    """Readable JSON with exactly one UAV object per line and one row per w line."""
    entries=[]
    for key,value in data.items():
        if key=='uavs':
            encoded='[\n'+',\n'.join('    '+json.dumps(c,ensure_ascii=False,allow_nan=False) for c in value)+'\n  ]'
        else:
            lines=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False).splitlines()
            encoded=lines[0]+''.join('\n  '+line for line in lines[1:])
        entries.append('  '+json.dumps(key)+': '+encoded)
    entries.append('  "grid": '+encoded_grid)
    return ('{\n'+',\n'.join(entries)+'\n}\n').encode('utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--calibrate-only',action='store_true',help='Choose budget fractions on val and update config only')
    args=parser.parse_args()
    if args.workers<1:
        parser.error('workers must be positive')
    config=load_config();source=read_source()
    if config['seeds']!=[42,43,44] or config['K_values']!=[6] or config['grid']!={'m':50,'n':50,'u':40}:
        raise ValueError('Expected agreed K=6, three seeds 42/43/44, 50x50, u=40')
    if args.calibrate_only:
        config=calibrate_budgets(config,source)
        CONFIG.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        return
    vector,scaled=weight_vectors(config)
    levels=config['budget_levels'];seeds=config['seeds']
    if len(levels)!=3 or any('fractions_by_K' not in level for level in levels):
        raise ValueError('Calibrate three budget levels before generation')
    designs=[{'K':K,'level':level['name'],'fraction':float(level['fractions_by_K'][str(K)]),
              'profile':f'K{K}_b{level["name"]}'}
             for K in config['K_values'] for level in levels]
    expected_per_image=len(designs)*len(seeds)
    base=args.output.resolve();base.mkdir(parents=True,exist_ok=True)
    config_hash=hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    if (base/'summary.json').exists() and json.loads((base/'summary.json').read_text(encoding='utf-8'))['config_sha256']!=config_hash:
        raise ValueError('Existing dataset uses another config; move it aside before generating')
    for split in ['val','test']:
        for d in designs:(base/split/d['profile']).mkdir(parents=True,exist_ok=True)
    schema=Draft202012Validator(json.loads((ROOT/'docs/instance.schema.json').read_text(encoding='utf-8')))

    def create(row):
        with Image.open(ROOT/row['merged_mask_path']) as im:
            im.load();raw=np.asarray(im);mask_hash=hashlib.sha256(im.tobytes()).hexdigest()
            numerators,crop=native_to_grid(raw,scaled)
        if mask_hash!=row['mask_sha256']:
            raise ValueError(f'Native mask differs from frozen source: {row["id"]}')
        w=weights_from_numerators(numerators)
        for i,j in [(0,0),(25,25),(49,49),(7,31),(43,12)]:
            pixels=raw[crop[1]+60*i:crop[1]+60*(i+1),crop[0]+60*j:crop[0]+60*(j+1)]
            if abs(np.bincount(pixels.ravel(),minlength=11)@vector/3600-w[i,j])>5.1e-13:
                raise ValueError('Independent native aggregation failed')
        fingerprint=hashlib.sha256(numerators.astype('<u4').tobytes()).hexdigest()
        grid={**config['grid'],'w':[[0 if v==0 else float(v) for v in line] for line in w]}
        encoded_grid=('{\n    "m": 50,\n    "n": 50,\n    "u": 40,\n    "w": [\n'+
                      ',\n'.join('      '+json.dumps(line,separators=(',',':'),allow_nan=False) for line in grid['w'])+
                      '\n    ]\n  }')
        total=float(w.sum());positive=int(np.count_nonzero(w));ii,jj=np.indices((50,50));records=[]
        for seed in seeds:
            depots=depots_for(row['id'],seed,6);depot=depots[0]
            for d in designs:
                B,D,reference=budget_parameters(config,d['K'],d['fraction'],depot)
                fleet=build_uavs(config,d['K'],B,depots)
                capacities=[craft['B'] for craft in fleet]
                name=f'rescuenet_{row["id"]}_50x50_K{d["K"]}_b{d["level"]}_s{seed}'
                data={'name':name,'source':{
                    'dataset':'RescueNet','image_id':row['id'],'split':row['split'],'original_split':row['original_split'],
                    'group_id':row['group_id'],'image_path':row['image_path'],'mask_path':row['mask_path'],
                    'seed':seed,'budget_level':d['level'],'budget_fraction':d['fraction'],
                    'budget_reference':reference,'budget_rule':config['budget_rule'],
                    'battery_multipliers':config['battery_allocation']['multipliers_by_K'][str(d['K'])],
                    'depot_policy':config['depot_allocation'],
                    'crop_xywh':crop,'cell_pixels':[60,60],'aggregation':'mean_pixel_rho',
                    'mask_sha256':mask_hash,'grid_sha256':fingerprint,'config_version':config['version'],
                    'zero_risk':total==0,'scale_assumption':'u=40m is a simulation scale, not georeferenced image scale'},
                    'K':d['K'],'D':D,'eta':config['eta'],
                    'uavs':fleet,'rho':config['rho']}
                validate_instance(dict(data,grid=grid))
                if row['id']==str(source.iloc[0].id):schema.validate(dict(data,grid=grid))
                payload=encode_instance(data,encoded_grid)
                relative=Path(row['split'])/d['profile']/f'{name}.json'
                write_identical_or_new(base/relative,payload)
                reachable=np.zeros((50,50),dtype=bool)
                for craft in fleet:
                    distance=40*np.hypot(ii-craft['depot'][0],jj-craft['depot'][1])
                    reachable|=(distance/craft['v_h']<=D+1e-9)&(2*craft['e_h']*distance+craft['eps']<=config['eta']*craft['B']+1e-9)
                records.append({'name':name,'path':relative.as_posix(),'image_id':row['id'],'split':row['split'],
                    'profile':d['profile'],'K':d['K'],'budget_level':d['level'],'budget_fraction':d['fraction'],
                    'B':B,'B_min':min(capacities),'B_max':max(capacities),'B_team':round(sum(capacities),6),
                    'B_values':json.dumps(capacities,separators=(',',':')),
                    'D':D,'seed':seed,'reference_depot_i':depot[0],'reference_depot_j':depot[1],
                    'depot_values':json.dumps(depots,separators=(',',':')),
                    'reference_energy':reference['energy'],'reference_time':reference['arrival_time'],
                    'total_risk':total,'positive_cells':positive,'zero_risk':total==0,
                    'reachable_positive_cells':int(np.count_nonzero(reachable&(w>0))),
                    'grid_sha256':fingerprint,'bytes':len(payload)})
        return records,total==0

    count=byte_count=zero_maps=0;by_split={};start=time.perf_counter()
    with (base/'manifest.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=None
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for image_no,(rows,zero) in enumerate(pool.map(create,source.to_dict('records')),1):
                if writer is None:writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader()
                writer.writerows(rows);count+=len(rows);zero_maps+=zero;byte_count+=sum(r['bytes'] for r in rows)
                by_split[rows[0]['split']]=by_split.get(rows[0]['split'],0)+len(rows)
                if image_no%250==0 or image_no==len(source):
                    handle.flush();print(f'{image_no}/{len(source)} images; {count:,} instances',flush=True)
    if count!=len(source)*expected_per_image:raise RuntimeError('Incomplete generation')
    summary={'config_sha256':config_hash,'config_file':'configs/rescuenet_instances.json',
        'source_manifest_file':'configs/rescuenet_split.csv',
        'source_manifest_sha256':hashlib.sha256(SOURCE_MANIFEST.read_bytes()).hexdigest(),
        'frozen_split_manifest_sha256':hashlib.sha256(SOURCE_MANIFEST.read_bytes()).hexdigest(),
        'source_images':len(source),'instances':count,'instances_per_image':expected_per_image,'seeds':seeds,
        'K_values':config['K_values'],'budget_levels':levels,'tree_weight':config['rho']['Tree'],
        'by_split':by_split,'zero_risk_maps':int(zero_maps),'json_bytes':byte_count,'elapsed_seconds':time.perf_counter()-start,
        'calibration':config['calibration'],'battery_allocation':config['battery_allocation'],
        'depot_allocation':config['depot_allocation'],
        'validation':{'generated_instances_checked':count,'native_cell_checks':len(source)*5,'schema_cases_checked':expected_per_image}}
    (base/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    level_table='\n'.join(f'| {d["K"]} | {d["level"]} | {100*d["fraction"]:g}% |' for d in designs)
    readme=f'''# RescueNet 50×50 — bộ instance K=6

**{count:,} instance** từ 4.494 ảnh, dùng cho bài toán khảo sát rủi ro với sáu UAV.
Mỗi ảnh có {expected_per_image} instance: 3 mức ngân sách × 3 seed (42, 43, 44).
Mỗi instance có sáu depot khác nhau và pin riêng cho từng UAV.

## Cấu trúc thư mục

```text
rescuenet_50x50/
  val/                       # {by_split['val']:,} instance, 899 ảnh
    K6_blow/
    K6_bmedium/
    K6_bhigh/
  test/                      # {by_split['test']:,} instance, 3.595 ảnh; cùng ba thư mục
  manifest.csv
  summary.json
  README.md
```

- **JSON:** input đầy đủ của một bài toán: lưới rủi ro, deadline, depot và pin của sáu UAV.
- **manifest.csv:** danh sách instance, đường dẫn tương đối, split, seed và ngân sách.
- **summary.json:** thông tin phiên bản, checksum và kết quả kiểm tra dữ liệu.

Ví dụ `rescuenet_10778_50x50_K6_blow_s42.json`: ảnh 10778, lưới 50×50, sáu UAV, mức low, seed 42.
low/medium/high là mức tài nguyên, không phải tỷ lệ rủi ro phủ.

## Sử dụng

1. Giải nén vào `data/instances/rescuenet_50x50/` trong repo, giữ cả manifest và summary.
2. Dùng val chọn tham số thuật toán; cố định tham số trước khi chạy test.
3. Đọc từng JSON và trả về sáu route theo thứ tự UAV trong `uavs`.

Không cần ảnh/mask gốc để giải bài toán. Cùng ảnh/seed giữ nguyên depot giữa ba mức ngân sách.
Đọc pin thực tại `uavs[k].B`; cột B trong manifest chỉ là pin cơ sở.
Có {zero_maps} ảnh có tổng rủi ro bằng 0: tỷ lệ phủ là N/A, không tính vào thống kê tỷ lệ.

Kiểm tra bộ dữ liệu từ thư mục gốc repo, sau khi cài `requirements-instances.txt`:

```powershell
python scripts/validate_instances.py --workers 6
```

Phương pháp sinh, trọng số và công thức ngân sách được trình bày trong
`docs/rescuenet_experiment_protocol.md` của repo; chuẩn input/nghiệm nằm trong `docs/instance_format.md`.
'''
    (base/'README.md').write_text(readme,encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='calibration'},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
