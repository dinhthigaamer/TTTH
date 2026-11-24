"""Generate RescueNet inputs using zig-zag-normalized energy/time budgets.

Only images with nine verified 10-50% optimum-coverage cases are written.
Unselected images are indexed separately; no exception instances are generated.
Use --grid-size 50 or 100. Source splits stay frozen.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image
from jsonschema import Draft202012Validator

from validate_instances import (ROOT,SOURCE_MANIFEST,load_config,validate_instance,
                                evaluate_routes,depots_for,budget_parameters,build_uavs,config_path_for)

SQUARE=3000


def native_to_grid(mask,rho_scaled,grid_size=50):
    if grid_size not in (50,100):
        raise ValueError('Supported grid sizes are 50 and 100')
    cell=SQUARE//grid_size
    a=np.asarray(mask)
    if a.ndim!=2 or a.dtype.kind not in 'iu' or a.min()<0 or a.max()>10:
        raise ValueError('Expected integer mask IDs 0..10')
    h,w=a.shape
    if min(h,w)<SQUARE:
        raise ValueError('Mask smaller than 3000x3000')
    x,y=(w-SQUARE)//2,(h-SQUARE)//2
    crop=a[y:y+SQUARE,x:x+SQUARE]
    numerator=rho_scaled[crop].reshape(grid_size,cell,grid_size,cell).sum(axis=(1,3),dtype=np.uint32)
    return numerator,[x,y,SQUARE,SQUARE]


def weights_from_numerators(numerators):
    if numerators.shape not in ((50,50),(100,100)):
        raise ValueError('Unexpected risk grid shape')
    denominator=20*(SQUARE//numerators.shape[0])**2
    return np.round(numerators.astype(float)/denominator,12)


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
    u=instance['grid']['u']
    w=np.asarray(instance['grid']['w']);cells=np.column_stack(np.nonzero(w>0))
    values=w[cells[:,0],cells[:,1]] if len(cells) else np.array([])
    available=np.ones(len(cells),dtype=bool);routes=[]
    for craft in instance['uavs']:
        cur=np.array(craft['depot']);route=[];flown=arrival=0.
        back=u*np.linalg.norm(cells-cur,axis=1)
        while available.any():
            distance=u*np.linalg.norm(cells-cur,axis=1)
            t=arrival+distance/craft['v_h']+(craft['tau'] if route else 0)
            e=craft['e_h']*(flown+distance+back)+(len(route)+1)*craft['eps']
            feasible=available&(t<=instance['D']+1e-9)&(e<=instance['eta']*craft['B']+1e-9)
            if not feasible.any():
                break
            best=int(np.argmax(np.where(feasible,values/(distance/u+1),-np.inf)))
            flown+=float(distance[best]);arrival=float(t[best]);cur=cells[best]
            route.append(cur.tolist());available[best]=False
        routes.append(route)
    return routes


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


def create_image(task):
    """Write nothing unless all nine cases have valid coverage certificates."""
    row,config,base=task
    base=Path(base);size=config['grid']['m'];cell=SQUARE//size
    vector,scaled=weight_vectors(config)
    with Image.open(ROOT/row['mask_path']) as im:
        im.load();raw=np.asarray(im);mask_hash=hashlib.sha256(im.tobytes()).hexdigest()
        numerators,crop=native_to_grid(raw,scaled,size)
    if mask_hash!=row['mask_sha256']:
        raise ValueError(f'Native mask differs from frozen source: {row["id"]}')
    w=weights_from_numerators(numerators)
    for i,j in [(0,0),(size//2,size//2),(size-1,size-1),(7,31),(43,12)]:
        pixels=raw[crop[1]+cell*i:crop[1]+cell*(i+1),crop[0]+cell*j:crop[0]+cell*(j+1)]
        if abs(np.bincount(pixels.ravel(),minlength=11)@vector/(cell*cell)-w[i,j])>5.1e-13:
            raise ValueError('Independent native aggregation failed')
    fingerprint=hashlib.sha256(numerators.astype('<u4').tobytes()).hexdigest()
    total=float(w.sum());positive=int(np.count_nonzero(w))
    from coverage_certificate import certify_seed,impossibility_certificate
    failures={};obstructions={};seed_designs={}
    for seed in config['seeds']:
        proof=impossibility_certificate(config,w,depots_for(row['id'],seed,6,size,size))
        if proof:obstructions[str(seed)]=proof;failures[str(seed)]=proof['reason']
    if not obstructions:
        for seed in config['seeds']:
            result,reason=certify_seed(config,w,depots_for(row['id'],seed,6,size,size))
            if result is None:
                import copy
                legacy=copy.deepcopy(config)
                legacy['coverage_certification'].update(upper_bound='reachable_top_weights_v1',upper_search_percent=49,upper_search_steps=18,
                    independent_energy_candidates=[],level_factor_candidates=[[.6,.8,1.],[.85,.925,1.],[.95,.975,1.]])
                result,reason=certify_seed(legacy,w,depots_for(row['id'],seed,6,size,size))
            if result is None:failures[str(seed)]=reason
            else:seed_designs[seed]=result
    if failures:
        excluded={'grid_size':size,'image_id':row['id'],'source_split':row['split'],
            'group_id':row['group_id'],'image_path':row['image_path'],'mask_path':row['mask_path'],
            'status':'proven_incompatible' if obstructions else 'not_certified',
            'reason_by_seed':json.dumps(failures,sort_keys=True,separators=(',',':')),
            'impossibility_proofs':json.dumps(obstructions,sort_keys=True,separators=(',',':')),
            'mask_sha256':mask_hash,'grid_sha256':fingerprint}
        return [],total==0,[],excluded
    grid={**config['grid'],'w':[[0 if v==0 else float(v) for v in line] for line in w]}
    encoded_grid=('{\n'+f'    "m": {size},\n    "n": {size},\n    "u": 40,\n    "w": [\n'+
        ',\n'.join('      '+json.dumps(line,separators=(',',':'),allow_nan=False) for line in grid['w'])+'\n    ]\n  }')
    ii,jj=np.indices((size,size));records=[];certificates=[]
    for seed in config['seeds']:
        depots=depots_for(row['id'],seed,6,size,size)
        for level,proof in zip(config['budget_levels'],seed_designs[seed]):
            alpha=proof['alpha'];time_alpha=proof.get('time_alpha',alpha)
            B,D,reference=budget_parameters(config,6,alpha,depots[0],time_alpha)
            fleet=build_uavs(config,6,B,depots);capacities=[c['B'] for c in fleet]
            name=f'rescuenet_{row["id"]}_{size}x{size}_K6_b{level["name"]}_s{seed}'
            data={'name':name,'source':{
                'dataset':'RescueNet','image_id':row['id'],'split':row['split'],'original_split':row['original_split'],
                'group_id':row['group_id'],'image_path':row['image_path'],'mask_path':row['mask_path'],
                'seed':seed,'budget_level':level['name'],'budget_fraction':alpha,'budget_time_fraction':time_alpha,
                'budget_reference':reference,'budget_rule':config['budget_rule'],
                'battery_multipliers':config['battery_allocation']['multipliers_by_K']['6'],
                'depot_policy':config['depot_allocation'],'crop_xywh':crop,'cell_pixels':[cell,cell],
                'aggregation':'mean_pixel_rho','mask_sha256':mask_hash,'grid_sha256':fingerprint,
                'config_version':config['version'],'case_set':'benchmark','zero_risk':False,
                'scale_assumption':'u=40m is a simulation scale, not georeferenced image scale'},
                'K':6,'D':D,'eta':config['eta'],'uavs':fleet,'rho':config['rho']}
            validate_instance(dict(data,grid=grid))
            payload=encode_instance(data,encoded_grid)
            relative=Path(row['split'])/f'K6_b{level["name"]}'/f'{name}.json'
            write_identical_or_new(base/relative,payload)
            certificates.append({'instance':name,'case_set':'benchmark','input_sha256':hashlib.sha256(payload).hexdigest(),
                **{k:proof[k] for k in ['lower_ticks','upper_ticks','total_ticks','max_cells_per_uav','method']},
                'solution':{'instance':name,'algorithm':'certificate_sweep_or_local_greedy_v1','routes':proof['routes']}})
            reachable=np.zeros((size,size),dtype=bool)
            for craft in fleet:
                distance=40*np.hypot(ii-craft['depot'][0],jj-craft['depot'][1])
                reachable|=(distance/craft['v_h']<=D+1e-9)&(2*craft['e_h']*distance+craft['eps']<=config['eta']*craft['B']+1e-9)
            records.append({'name':name,'path':relative.as_posix(),'image_id':row['id'],'split':row['split'],
                'profile':f'K6_b{level["name"]}','K':6,'budget_level':level['name'],'budget_fraction':alpha,
                'budget_time_fraction':time_alpha,'B':B,'B_min':min(capacities),'B_max':max(capacities),
                'B_team':round(sum(capacities),6),'B_values':json.dumps(capacities,separators=(',',':')),
                'D':D,'seed':seed,'reference_depot_i':depots[0][0],'reference_depot_j':depots[0][1],
                'depot_values':json.dumps(depots,separators=(',',':')),'reference_energy':reference['energy'],
                'reference_time':reference['arrival_time'],'total_risk':total,'positive_cells':positive,'zero_risk':False,
                'reachable_positive_cells':int(np.count_nonzero(reachable&(w>0))),'grid_sha256':fingerprint,'bytes':len(payload),
                'case_set':'benchmark','certified_lower_ratio':proof['lower_ticks']/proof['total_ticks'],
                'certified_upper_ratio':proof['upper_ticks']/proof['total_ticks']})
    return records,False,certificates,None


def dataset_readme(size,summary):
    return f'''# RescueNet {size}×{size}

Bộ dữ liệu chỉ chứa **{summary['instances']:,} instance đã xác thực** từ {summary['source_images']:,} ảnh.
K=6; sáu depot khác nhau; ba mức ngân sách và ba seed 42/43/44, tổng 9 instance/ảnh.
Val: **{summary['by_split'].get('val',0):,}**; test: **{summary['by_split'].get('test',0):,}**.
Mỗi instance có nghiệm khả thi đạt ≥10% và cận trên chứng minh mức phủ tối ưu ≤50%.

```text
rescuenet_{size}x{size}/
  val/                       # K6_blow, K6_bmedium, K6_bhigh
  test/                      # cùng ba thư mục cấu hình
  manifest.csv
  certificates.jsonl
  summary.json
  README.md
```

- JSON chứa đầy đủ input theo `docs/instance_format.md`.
- Manifest liệt kê các instance; certificates lưu nghiệm và cận kiểm chứng; summary lưu phiên bản và checksum.
- Dùng val chọn tham số, cố định tham số trước khi đánh giá test.
- Không có edge instance. Ảnh không được chọn chỉ được ghi chỉ mục nguồn riêng trong `data/rescuenet_unselected/`.

Giải nén vào thư mục trống `data/instances/rescuenet_{size}x{size}/`, không chồng lên bản cũ.
Chạy kiểm tra từ gốc repo:

```powershell
python scripts/validate_instances.py data/instances/rescuenet_{size}x{size} --workers 6
```

Phương pháp sinh và công thức: `docs/rescuenet_experiment_protocol.md` trong repo.
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid-size',type=int,choices=[50,100],default=50)
    parser.add_argument('--output',type=Path,default=None)
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    if args.workers<1:parser.error('workers must be positive')
    size=args.grid_size;config_path=config_path_for(size);config=load_config(config_path);source=read_source()
    if (config['seeds']!=[42,43,44] or config['K_values']!=[6] or config['grid']!={'m':size,'n':size,'u':40}
            or not config.get('selection',{}).get('only_certified')):
        raise ValueError('Expected K6, three fixed seeds, selected grid and certified-only selection')
    base=(args.output or ROOT/f'data/instances/rescuenet_{size}x{size}').resolve();base.mkdir(parents=True,exist_ok=True)
    config_hash=hashlib.sha256(config_path.read_bytes()).hexdigest()
    if (base/'summary.json').exists() and json.loads((base/'summary.json').read_text(encoding='utf-8'))['config_sha256']!=config_hash:
        raise ValueError('Existing dataset uses another config; move it aside before generating')
    for split in ['val','test']:
        for level in config['budget_levels']:(base/split/f'K6_b{level["name"]}').mkdir(parents=True,exist_ok=True)
    rejected_root=ROOT/'data/rescuenet_unselected';rejected_root.mkdir(parents=True,exist_ok=True)
    rejected_path=rejected_root/f'{size}x{size}.csv'
    fields=['grid_size','image_id','source_split','group_id','image_path','mask_path','status','reason_by_seed','impossibility_proofs','mask_sha256','grid_sha256']
    count=byte_count=zero_maps=selected=excluded=0;by_split={};reasons={};start=time.perf_counter();schema_cases=0
    schema=Draft202012Validator(json.loads((ROOT/'docs/instance.schema.json').read_text(encoding='utf-8')))
    with (base/'manifest.csv').open('w',encoding='utf-8-sig',newline='') as manifest, (base/'certificates.jsonl').open('w',encoding='utf-8') as certificates, rejected_path.open('w',encoding='utf-8-sig',newline='') as rejects:
        writer=None;reject_writer=csv.DictWriter(rejects,fieldnames=fields);reject_writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            tasks=((row,config,str(base)) for row in source.to_dict('records'))
            for image_no,(rows,zero,proofs,rejected) in enumerate(pool.map(create_image,tasks,chunksize=1),1):
                zero_maps+=zero
                if rejected:
                    reject_writer.writerow(rejected);excluded+=1
                    reasons[rejected['status']]=reasons.get(rejected['status'],0)+1
                else:
                    if writer is None:
                        writer=csv.DictWriter(manifest,fieldnames=list(rows[0]));writer.writeheader()
                        for row in rows:schema.validate(json.loads((base/row['path']).read_bytes()));schema_cases+=1
                    writer.writerows(rows)
                    for proof in proofs:certificates.write(json.dumps(proof,separators=(',',':'),allow_nan=False)+'\n')
                    selected+=1;count+=len(rows);byte_count+=sum(r['bytes'] for r in rows)
                    split=rows[0]['split'];by_split[split]=by_split.get(split,0)+len(rows)
                if image_no%250==0 or image_no==len(source):
                    manifest.flush();rejects.flush();certificates.flush()
                    print(f'{image_no}/{len(source)} images; {count:,} certified instances; {excluded} unselected sources',flush=True)
    if not count or count!=selected*9 or selected+excluded!=len(source):raise RuntimeError('Incomplete selection')
    summary={'config_sha256':config_hash,'config_file':config_path.relative_to(ROOT).as_posix(),
        'source_manifest_file':'configs/rescuenet_split.csv','source_manifest_sha256':hashlib.sha256(SOURCE_MANIFEST.read_bytes()).hexdigest(),
        'frozen_split_manifest_sha256':hashlib.sha256(SOURCE_MANIFEST.read_bytes()).hexdigest(),
        'source_images_scanned':len(source),'source_images':selected,'unselected_source_images':excluded,
        'unselected_catalog_file':rejected_path.relative_to(ROOT).as_posix(),'unselected_catalog_sha256':hashlib.sha256(rejected_path.read_bytes()).hexdigest(),
        'unselected_by_status':reasons,'instances':count,'instances_per_image':9,'seeds':config['seeds'],'K_values':[6],
        'grid':config['grid'],'budget_levels':config['budget_levels'],'by_split':by_split,'zero_risk_maps':0,
        'zero_risk_sources_scanned':int(zero_maps),'json_bytes':byte_count,'elapsed_seconds':time.perf_counter()-start,
        'battery_allocation':config['battery_allocation'],'depot_allocation':config['depot_allocation'],
        'by_case_set':{'benchmark':count},'coverage_certification':config['coverage_certification'],
        'certificates_sha256':hashlib.sha256((base/'certificates.jsonl').read_bytes()).hexdigest(),
        'validation':{'generated_instances_checked':count,'native_cell_checks':len(source)*5,'schema_cases_checked':schema_cases}}
    (base/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (base/'README.md').write_text(dataset_readme(size,summary),encoding='utf-8')
    print(json.dumps({'grid_size':size,'instances':count,'selected_images':selected,'unselected_images':excluded,'by_split':by_split},indent=2),flush=True)


if __name__=='__main__':main()
