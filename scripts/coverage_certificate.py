"""Constructive lower bound and conservative upper bound on optimal risk.

Weights are exact integers in 1e-12 units (the input JSON precision).
Every feasible route visits only singly reachable cells. Distinct consecutive
grid cells cost at least u metres, so time and energy bound the route length.
Relaxing route geometry and duplicate assignments then yields valid upper bounds.
"""
import math
from functools import lru_cache

import numpy as np


TICKS=10**12


@lru_cache(maxsize=64)
def positive_step(mask,m,n,u):
    occupied=np.frombuffer(mask,dtype=bool).reshape(m,n)
    for squared,dx,dy in ordered_offsets(m,n):
        if (occupied[dx:,dy:]&occupied[:m-dx,:n-dy]).any() or (dx and dy and (occupied[dx:,:n-dy]&occupied[:m-dx,dy:]).any()):
            return max(u,u*math.sqrt(squared)-1e-7)
    return u


@lru_cache(maxsize=4)
def ordered_offsets(m,n):
    return sorted((dx*dx+dy*dy,dx,dy) for dx in range(m) for dy in range(n) if dx or dy)


class CoverageBound:
    def __init__(self,w,u,depots,method='reachable_top_weights_v1'):
        w=np.asarray(w,dtype=float)
        if not np.array_equal(w,np.round(w,12)):
            raise ValueError('Certified weights must have at most 12 decimal places')
        self.shape=w.shape
        self.u=u
        self.method=method
        if method not in ('reachable_top_weights_v1','entry_distance_top_weights_v2'):
            raise ValueError('Unknown upper-bound method')
        ticks=np.rint(np.asarray(w)*TICKS).astype(np.int64)
        self.total=int(ticks.sum())
        cells=np.column_stack(np.nonzero(ticks>0))
        values=ticks[cells[:,0],cells[:,1]]
        order=np.argsort(-values,kind='stable')
        self.cells=cells[order]
        self.values=values[order]
        self.distances=[u*np.linalg.norm(self.cells-np.asarray(d),axis=1) for d in depots]
        self.minimum_step=u
        if method=='entry_distance_top_weights_v2' and len(cells)>1:
            m,n=w.shape
            self.minimum_step=positive_step((ticks>0).tobytes(),m,n,u)

    def upper(self,instance):
        union=np.zeros(len(self.values),dtype=bool)
        counts=[];individual=[]
        for craft,distance in zip(instance['uavs'],self.distances):
            entry=max(0.,float(distance.min())-1e-7) if len(distance) and self.method=='entry_distance_top_weights_v2' else 0.
            gap=self.minimum_step/craft['v_h']+craft['tau']
            count_t=math.floor(math.nextafter((instance['D']+1e-9-entry/craft['v_h'])/gap,math.inf))+1
            step=craft['e_h']*self.minimum_step
            count_e=math.floor(math.nextafter((instance['eta']*craft['B']+1e-9+step-2*craft['e_h']*entry)/(step+craft['eps']),math.inf))
            count=max(0,min(math.prod(self.shape),count_t,count_e))
            radius=min((instance['D']+1e-9)*craft['v_h'],
                       (instance['eta']*craft['B']+1e-9-craft['eps'])/(2*craft['e_h']))
            # Outward slack only enlarges the relaxation (never excludes an
            # actually reachable cell due to square-root/float rounding).
            reachable=distance<=radius+1e-7
            union|=reachable
            counts.append(count)
            individual.append(int(self.values[reachable][:count].sum()))
        pooled=int(self.values[union][:sum(counts)].sum())
        bound=min(self.total,pooled,sum(individual))
        return {'upper_ticks':bound,'total_ticks':self.total,
                'max_cells_per_uav':counts,'method':self.method}


def route_ticks(instance,routes):
    cells={tuple(c) for route in routes for c in route}
    return sum(round(instance['grid']['w'][i][j]*TICKS) for i,j in cells)


def impossibility_certificate(config,w,depots):
    """Sufficient proofs of impossibility, never a heuristic failure label."""
    ticks=np.rint(np.asarray(w)*TICKS).astype(np.int64)
    total=int(ticks.sum())
    if not total:return {'reason':'zero_risk','total_ticks':0}
    if int(ticks[ticks*2<=total].sum())*10<total:
        return {'reason':'discrete_risk_gap','total_ticks':total,
                'at_most_half_cell_ticks':int(ticks[ticks*2<=total].sum())}
    cells=np.column_stack(np.nonzero(ticks>0))
    if len(cells)!=2:return None
    values=ticks[cells[:,0],cells[:,1]]
    order=np.argsort(values)
    minor,dominant=cells[order[0]],cells[order[1]]
    if int(values[order[1]])*2<=total:return None
    craft=config['uav'];u=config['grid']['u'];eta=config['eta']
    factors=np.asarray(config['battery_allocation']['multipliers_by_K']['6'])
    ds=u*np.linalg.norm(np.asarray(depots)-minor,axis=1)
    db=u*np.linalg.norm(np.asarray(depots)-dominant,axis=1)
    # First five rounded capacities differ from a_k*B_base by <=0.5e-6;
    # the final residual differs by <=2.5e-6. Use 3e-6 in both directions.
    delta=3e-6;blockers=[]
    for k in range(6):
        required=(2*craft['e_h']*ds[k]+craft['eps'])/eta
        base_lower=max(0.,(required-delta)/factors[k])
        candidates=[j for j in range(6)
            if db[j]/craft['v_h']<ds[k]/craft['v_h']-1e-8
            and eta*(factors[j]*base_lower-delta)>2*craft['e_h']*db[j]+craft['eps']+1e-7]
        if not candidates:return None
        blockers.append(candidates[0])
    return {'reason':'dominant_cell_forced_by_minor_reachability','total_ticks':total,
            'minor_cell':minor.tolist(),'dominant_cell':dominant.tolist(),
            'blocker_uavs':blockers,'capacity_rounding_slack':delta}


@lru_cache(maxsize=4)
def sweep_orders(m,n):
    orders=[]
    for transpose in (False,True):
        for flip_i in (False,True):
            for flip_j in (False,True):
                cells=np.arange(m*n).reshape(m,n)
                if transpose:cells=cells.T
                if flip_i:cells=cells[::-1]
                if flip_j:cells=cells[:,::-1]
                cells=cells.copy();cells[1::2]=cells[1::2,::-1]
                order=cells.ravel();inverse=np.empty(m*n,dtype=int)
                inverse[order]=np.arange(m*n)
                orders.append((order,inverse))
    return orders


def sweep_routes(instance):
    """Choose feasible prefixes of eight sweeps, starting at each own depot."""
    w=np.asarray(instance['grid']['w']);m,n=w.shape;u=instance['grid']['u']
    values=w.ravel();available=values>0;routes=[]
    for craft in instance['uavs']:
        depot=np.asarray(craft['depot']);best_gain=-1.;best=np.array([],dtype=int)
        for order,inverse in sweep_orders(m,n):
            start=inverse[depot[0]*n+depot[1]]
            sequence=np.concatenate((order[start:],order[:start]))
            sequence=sequence[available[sequence]]
            if not len(sequence):continue
            cells=np.column_stack((sequence//n,sequence%n))
            travel=np.diff(cells,axis=0,prepend=depot[None,:])
            flown=np.cumsum(u*np.hypot(travel[:,0],travel[:,1]))
            back=u*np.hypot(cells[:,0]-depot[0],cells[:,1]-depot[1])
            L=np.arange(1,len(sequence)+1)
            times=flown/craft['v_h']+(L-1)*craft['tau']
            energy=craft['e_h']*(flown+back)+L*craft['eps']
            feasible=np.flatnonzero((times<=instance['D']-1e-7)&(energy<=instance['eta']*craft['B']-1e-7))
            if not len(feasible):continue
            prefix=sequence[:feasible[-1]+1];gain=float(values[prefix].sum())
            if gain>best_gain:best_gain=gain;best=prefix
        available[best]=False
        routes.append(np.column_stack((best//n,best%n)).tolist())
    return routes


def local_greedy_routes(instance):
    """Fast constructive heuristic; only feasibility matters to the proof.

    Search a local 9x9 window and 32 highest-risk remaining cells; if neither
    offers a feasible extension, search all remaining positive cells once.
    """
    w=np.asarray(instance['grid']['w']);m,n=w.shape;u=instance['grid']['u']
    values=w.ravel();available=values>0
    ranked=np.argsort(-values,kind='stable')
    ranked=ranked[values[ranked]>0]
    ii,jj=np.indices((m,n));ii=ii.ravel();jj=jj.ravel()
    di,dj=np.meshgrid(np.arange(-4,5),np.arange(-4,5),indexing='ij')
    di=di.ravel();dj=dj.ravel()
    table=u*np.hypot(np.arange(m)[:,None],np.arange(n)[None,:])
    routes=[]
    for craft in instance['uavs']:
        ci,cj=craft['depot'];route=[];flown=arrival=0.
        back=table[np.abs(ii-ci),np.abs(jj-cj)]
        while available.any():
            top=ranked[available[ranked]][:32]
            ri,rj=ci+di,cj+dj;valid=(ri>=0)&(ri<m)&(rj>=0)&(rj<n)
            local=ri[valid]*n+rj[valid]
            candidates=np.unique(np.concatenate((local[available[local]],top)))
            chosen=None
            for pool in (candidates,np.flatnonzero(available)):
                distance=table[np.abs(ii[pool]-ci),np.abs(jj[pool]-cj)]
                times=arrival+distance/craft['v_h']+(craft['tau'] if route else 0)
                energy=craft['e_h']*(flown+distance+back[pool])+(len(route)+1)*craft['eps']
                feasible=(times<=instance['D']-1e-7)&(energy<=instance['eta']*craft['B']-1e-7)
                if feasible.any():
                    best=int(np.argmax(np.where(feasible,values[pool]/(distance/u+1),-np.inf)))
                    chosen=int(pool[best]);flown+=float(distance[best]);arrival=float(times[best])
                    break
            if chosen is None:
                break
            ci,cj=int(ii[chosen]),int(jj[chosen]);route.append([ci,cj]);available[chosen]=False
        routes.append(route)
    return routes


def certify_seed(config,w,depots):
    """Fixed per-instance budget construction; no optimizer tuning on test."""
    from validate_instances import budget_parameters,build_uavs,evaluate_routes
    settings=config['coverage_certification']
    obstruction=impossibility_certificate(config,w,depots)
    if obstruction:return None,obstruction['reason']
    bound=CoverageBound(w,config['grid']['u'],depots,settings['upper_bound'])
    if not bound.total:
        return None,'zero_risk'
    if int(bound.values[bound.values*2<=bound.total].sum())*10<bound.total:
        return None,'discrete_risk_gap'
    grid={**config['grid'],'w':w.tolist()}
    def instance(alpha,time_alpha=None):
        B,D,_=budget_parameters(config,6,alpha,depots[0],time_alpha)
        return {'grid':grid,'K':6,'D':D,'eta':config['eta'],
                'uavs':build_uavs(config,6,B,depots)}
    def upper_search(make):
        lo,hi=settings['alpha_min'],settings['alpha_max']
        if bound.upper(make(lo))['upper_ticks']*100>bound.total*settings['upper_search_percent']:
            return None
        for _ in range(settings['upper_search_steps']):
            mid=(lo+hi)/2
            if bound.upper(make(mid))['upper_ticks']*100<=bound.total*settings['upper_search_percent']:
                lo=mid
            else:hi=mid
        return lo
    def levels(high,energy_alpha=None):
        for heuristic in (sweep_routes,local_greedy_routes):
            cache={}
            for factors in settings['level_factor_candidates']:
                designs=[]
                for factor in factors:
                    time_alpha=round(high*factor,12)
                    alpha=time_alpha if energy_alpha is None else energy_alpha
                    if time_alpha not in cache:
                        x=instance(alpha,time_alpha);upper=bound.upper(x)
                        routes=heuristic(x);evaluate_routes(x,routes)
                        cache[time_alpha]=(routes,route_ticks(x,routes),upper)
                    routes,lower,upper=cache[time_alpha]
                    if lower*10<bound.total or upper['upper_ticks']*2>bound.total:break
                    designs.append({'alpha':alpha,'time_alpha':time_alpha,'routes':routes,'lower_ticks':lower,**upper})
                if len(designs)==3 and designs[0]['time_alpha']<designs[1]['time_alpha']<designs[2]['time_alpha']:
                    pairs={budget_parameters(config,6,d['alpha'],depots[0],d['time_alpha'])[:2] for d in designs}
                    if len(pairs)==3:return designs
        return None
    high=upper_search(instance)
    if high is not None:
        result=levels(high)
        if result:return result,None
    for energy_alpha in settings.get('independent_energy_candidates',[]):
        high=upper_search(lambda time_alpha:instance(energy_alpha,time_alpha))
        if high is not None:
            result=levels(high,energy_alpha)
            if result:return result,None
    return None,'no_three_certified_levels_found'
