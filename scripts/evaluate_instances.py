"""Measure achieved risk coverage of every generated instance using frozen greedy.

Outputs a result CSV and summary JSON/Markdown in a separate directory per grid size.
Inputs, budgets, weights, and source splits are never changed. No optimality claim.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from generate_rescuenet_instances import greedy_routes
from validate_instances import ROOT,DEFAULT_DATA,evaluate_routes,dataset_config_path,load_config


def evaluate_image(task):
    base,records=task
    results=[]
    for row in records:
        instance=json.loads((Path(base)/row['path']).read_bytes())
        start=time.perf_counter()
        routes=greedy_routes(instance)
        seconds=time.perf_counter()-start
        score=evaluate_routes(instance,routes)
        total=float(np.asarray(instance['grid']['w']).sum())
        if abs(total-row['total_risk'])>1e-8:
            raise ValueError('Risk sum differs from manifest')
        coverage=score['risk_coverage']
        if coverage is not None:
            if not -1e-10<=coverage<=1+1e-10:
                raise ValueError('Invalid coverage')
            coverage=float(np.clip(coverage,0,1))
        results.append({'name':row['name'],'path':row['path'],'image_id':row['image_id'],
            'split':row['split'],'K':row['K'],'budget_level':row['budget_level'],'seed':row['seed'],
            'B_mean_per_uav':row['B'],'B_min':min(c['B'] for c in instance['uavs']),
            'B_max':max(c['B'] for c in instance['uavs']),
            'B_team':round(sum(c['B'] for c in instance['uavs']),6),
            'D':row['D'],'total_risk':total,'covered_risk':score['objective'],
            'risk_coverage':coverage,'cover_rate':score['cover_rate'],
            'observed_cells':int(round(score['cover_rate']*instance['grid']['m']*instance['grid']['n'])),
            'max_arrival_seconds':max(s['arrival_time'] for s in score['uavs']),
            'max_energy_fraction':max(s['energy']/(instance['eta']*craft['B'])
                                       for s,craft in zip(score['uavs'],instance['uavs'])),
            'greedy_seconds':seconds,'all_routes_feasible':True})
    return results


def group_summary(frame,keys):
    output=[]
    for values,sub in frame.groupby(keys,sort=True):
        if not isinstance(values,tuple):values=(values,)
        values=tuple(v.item() if isinstance(v,np.generic) else v for v in values)
        positive=sub[sub.total_risk>0]
        # First average seeds within an image. The image remains the unit of
        # aggregation; the three depot variants are not independent scenes.
        image_rates=positive.groupby('image_id').risk_coverage.mean()
        ratios=positive.risk_coverage
        output.append({**dict(zip(keys,values)),
            'instances':len(sub),'positive_instances':len(positive),'zero_risk_instances':len(sub)-len(positive),
            'positive_images':int(positive.image_id.nunique()),
            'mean_coverage_pct':float(100*image_rates.mean()) if len(image_rates) else None,
            'median_image_coverage_pct':float(100*image_rates.median()) if len(image_rates) else None,
            'median_instance_coverage_pct':float(100*ratios.median()) if len(ratios) else None,
            'q10_image_coverage_pct':float(100*image_rates.quantile(.1)) if len(image_rates) else None,
            'q90_image_coverage_pct':float(100*image_rates.quantile(.9)) if len(image_rates) else None,
            'share_instances_10_to_50_pct':float(100*ratios.between(.1,.5).mean()) if len(ratios) else None,
            'share_instances_full_pct':float(100*(ratios>=1-1e-9).mean()) if len(ratios) else None,
            'pooled_covered_risk_pct':float(100*positive.covered_risk.sum()/positive.total_risk.sum()) if len(positive) else None})
    return output


def main():
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--output',type=Path,default=None)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--summarize-only',action='store_true',help='Rebuild reports from a complete existing results.csv')
    args=parser.parse_args()
    if args.workers<1:parser.error('workers must be positive')
    base=args.data.resolve()
    manifest=pd.read_csv(base/'manifest.csv',dtype={'image_id':str})
    metadata=json.loads((base/'summary.json').read_text(encoding='utf-8'))
    config_path=dataset_config_path(metadata);config=load_config(config_path)
    config_hash=hashlib.sha256(config_path.read_bytes()).hexdigest()
    if config_hash!=metadata['config_sha256'] or len(manifest)!=metadata['instances']:
        raise ValueError('Dataset/config mismatch')
    if 'coverage_certification' in config:
        manifest=manifest[manifest.case_set=='benchmark'].copy()
    size=config['grid']['m']
    out=args.output or ROOT/('outputs/rescuenet_coverage' if size==50 else f'outputs/rescuenet_coverage_{size}x{size}')
    out.mkdir(parents=True,exist_ok=True)
    previous_summary=None
    if args.summarize_only and (out/'summary.json').exists():
        previous_summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
        if previous_summary['config_sha256']!=config_hash:
            raise ValueError('Existing evaluation results belong to another configuration')
    start=time.perf_counter();count=0
    if not args.summarize_only:
        tasks=[(str(base),sub.to_dict('records')) for _,sub in manifest.groupby('image_id',sort=True)]
        with (out/'results.csv').open('w',encoding='utf-8-sig',newline='') as handle:
            writer=None
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for n,rows in enumerate(pool.map(evaluate_image,tasks,chunksize=2),1):
                    if writer is None:writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader()
                    writer.writerows(rows);count+=len(rows)
                    if n%250==0 or n==len(tasks):
                        handle.flush();print(f'Evaluated {n}/{len(tasks)} images, {count:,} instances; '
                                            f'{(time.perf_counter()-start)/60:.1f} min',flush=True)
    df=pd.read_csv(out/'results.csv',dtype={'image_id':str})
    if (len(df)!=len(manifest) or set(df.name)!=set(manifest.name) or df.name.duplicated().any()
            or not df.all_routes_feasible.all()):
        raise ValueError('Incomplete/invalid evaluation')
    expected=manifest.set_index('name').reindex(df.name)
    for column in ['image_id','split','K','budget_level','seed']:
        if not np.array_equal(df[column].to_numpy(),expected[column].to_numpy()):
            raise ValueError(f'Evaluation results differ from manifest: {column}')
    for actual,column in [('B_mean_per_uav','B'),('B_min','B_min'),('B_max','B_max'),('B_team','B_team'),
                          ('D','D'),('total_risk','total_risk')]:
        if not np.allclose(df[actual],expected[column],rtol=0,atol=1e-8):
            raise ValueError(f'Evaluation results use different instance parameters: {column}')
    count=len(df)
    all_rows=df.assign(split='all')
    summary={'algorithm':'greedy gain/(distance_in_cells+1), sequential UAVs with shared covered set',
        'meaning':'feasible achieved coverage, not an optimum or an upper bound',
        'battery_allocation':config['battery_allocation'],
        'depot_allocation':config['depot_allocation'],
        'config_sha256':config_hash,
        'greedy_code_sha256':previous_summary['greedy_code_sha256'] if previous_summary else
                            hashlib.sha256((ROOT/'scripts/generate_rescuenet_instances.py').read_bytes()).hexdigest(),
        'instances':count,'source_images':int(df.image_id.nunique()),'positive_instances':int((df.total_risk>0).sum()),
        'zero_risk_instances':int((df.total_risk==0).sum()),'all_routes_independently_feasible':True,
        'elapsed_seconds':time.perf_counter()-start if not args.summarize_only else
                          previous_summary.get('elapsed_seconds') if previous_summary else None,
        'aggregation':'average seeds within each positive-risk image, then summarize images; zero-risk ratios are N/A',
        'by_split_K_level':group_summary(df,['split','K','budget_level']),
        'all_K_level':group_summary(all_rows,['split','K','budget_level']),
        'by_split_K_level_seed':group_summary(df,['split','K','budget_level','seed'])}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# Mức phủ rủi ro thực tế bằng greedy','',
        f'Đã chạy **{count:,} instance** từ **{df.image_id.nunique()} ảnh**; mọi route được kiểm tra độc lập.',
        'Đánh giá với cấu hình pin hiện tại đã đóng băng; không hiệu chỉnh tham số theo kết quả của lượt chạy này.',
        'Đây là kết quả greedy, không phải nghiệm tối ưu hoặc mức phủ bắt buộc của instance.','',
        'Tỉ lệ mỗi instance: tổng w của các ô được quan sát ít nhất một lần / tổng w toàn bản đồ.',
        'Trước khi tổng hợp, lấy trung bình ba seed trong từng ảnh. Chỉ tính tỷ lệ cho ảnh có tổng w>0.',
        'Bản đồ toàn 0 có mục tiêu tuyệt đối 0 và tỷ lệ N/A; không cộng chúng như tỷ lệ 0%.','']
    for split,label in [('test','Test'),('val','Val'),('all','Toàn bộ')]:
        rows=(summary['all_K_level'] if split=='all' else summary['by_split_K_level'])
        rows=[r for r in rows if r['split']==split]
        lines.extend([f'## {label}','','| K | Mức | Trung bình (%) | Trung vị theo ảnh (%) | Instance trong 10–50% (%) |',
                      '|---:|---|---:|---:|---:|'])
        for K in sorted(df.K.unique()):
            for level in ['low','medium','high']:
                r=next(r for r in rows if r['K']==K and r['budget_level']==level)
                lines.append(f'| {K} | {level} | {r["mean_coverage_pct"]:.2f} | {r["median_image_coverage_pct"]:.2f} | {r["share_instances_10_to_50_pct"]:.2f} |')
        lines.append('')
    lines.extend(['`results.csv`: kết quả từng instance; `summary.json`: thống kê và checksum.',
                  'Test đã được đánh giá với cấu hình cố định; không dùng bảng này để chỉnh tham số rồi vẫn coi test là chưa được xem.'])
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print((out/'README.md').read_text(encoding='utf-8'),flush=True)


if __name__=='__main__':main()
