import pandas as pd, numpy as np, json, os
from scipy import stats
os.chdir('/root/.claude/uploads/dd81ba2d-04ad-5f4d-8eea-b65adeafd9d9/')
OUT='/tmp/claude-0/-home-user-quest-plutchik-VAS/dd81ba2d-04ad-5f4d-8eea-b65adeafd9d9/scratchpad/report/data.json'
def f(x,d=2):
    return None if pd.isna(x) else round(float(x),d)
def pf(p):
    if pd.isna(p): return ''
    return '<.001' if p<.001 else f"{p:.3f}".lstrip('0')
D={}
P0=pd.read_csv('141c89a3-P0_distributions.csv'); m=P0[P0.method=='mean_flip']
w=m.pivot_table(index='measure',columns='data',values=['skew','shapiro_p','max_abs_z'])
order=["LZC","SampEn_m2","SpectralSlope","IRASA_slope","ACW50","IRASA_oscpower","SpectralEntropy","ACW0","MedianFreq","Spec_lowhigh_8Hz","MSE_costa_m2_AUC","ACF_area_to_zero"]
D['dist']=[[k,f(w.loc[k,('skew','raw')]),pf(w.loc[k,('shapiro_p','raw')]),f(w.loc[k,('max_abs_z','raw')],1),f(w.loc[k,('skew','transformed')]),pf(w.loc[k,('shapiro_p','transformed')]),f(w.loc[k,('max_abs_z','transformed')],1)] for k in order]
PA=pd.read_csv('06be692c-P1_parallel_analysis.csv')
D['parallel']=[[int(r.component),f(r.eigenvalue),f(r.random_95)] for r in PA[PA.method=='mean_flip'].head(6).itertuples()]
D['parallel_pca']=[[int(r.component),f(r.eigenvalue),f(r.random_95)] for r in PA[PA.method=='pca_flip'].head(6).itertuples()]
L=pd.read_csv('92c3f588-P1_loadings.csv'); lw=L.pivot_table(index='measure',columns=['method','factor'],values='loading')
D['loadings']=[[mm,f(lw.loc[mm,('mean_flip','F1')]),f(lw.loc[mm,('mean_flip','F2')]),f(lw.loc[mm,('pca_flip','F1')]),f(lw.loc[mm,('pca_flip','F2')])] for mm in order]
C=pd.read_csv('59f411a6-P1_congruence.csv'); D['congruence']=[[r.method,r.factor,r.compared_with,f(r.congruence,3)] for r in C.itertuples()]
F=pd.read_csv('f3f9d358-P1_factor_scores.csv',dtype={'subj':str}); st=[]
for meth,g in F.groupby('method'):
    w2=g.pivot_table(index='subj',columns='condition',values=['F1','F2'])
    for fac in ['F1','F2']:
        row=[meth,fac]
        for c in ['sad','fear']:
            d=w2[[(fac,'baseline'),(fac,c)]].dropna(); row.append(f(stats.spearmanr(d.iloc[:,0],d.iloc[:,1])[0]))
        for c in ['sad','fear']:
            s=(w2[(fac,c)]-w2[(fac,'baseline')]).dropna(); row+= [f(s.mean()), pf(stats.wilcoxon(s)[1])]
        st.append(row)
D['stability']=st
B=pd.read_csv('8f73ef6b-P1_factor_vs_behaviour.csv')
names={'latent_sd':'Latent SD (partial on mean)','sd':'SD','mean':'Mean level','frac_nonzero':'Share non-zero','first_minus_rest':'First − rest'}
D['fb']=[[r.method,r.condition,r.factor,names[r.behaviour],int(r.n),f(r.rho),pf(r.p)] for r in B.itertuples()]
D['fb_count']={m:[int(((B.method==m)&(B.p<.05)).sum()),int((B.method==m).sum())] for m in ['mean_flip','pca_flip']}
V=pd.read_csv('d0eef283-V2_exploratory.csv')
for h in ['H2','H4']:
    sub=V[(V.hyp==h)&(V.method=='mean_flip')]
    g=sub.pivot_table(index='measure',columns='group',values='stat'); p=sub.pivot_table(index='measure',columns='group',values='p')
    D['v2_'+h]={'groups':list(g.columns),'rows':[[mm]+[[f(g.loc[mm,c]),pf(p.loc[mm,c])] for c in g.columns] for mm in order]}
    D['v2_'+h+'_count']={mt:int(((V.hyp==h)&(V.method==mt)&(V.p<.05)).sum()) for mt in ['mean_flip','pca_flip']}
V1=pd.read_csv('cfc868aa-V1_primary.csv'); D['v1']=[[r.hyp,r.label,r.group,r.measure,int(r.n),f(r.stat),pf(r.p),pf(r.p_holm)] for r in V1.itertuples()]
D['v2_counts']={h:[int(((V.hyp==h)&(V.p<.05)).sum()),int(V[V.hyp==h].p.notna().sum())] for h in sorted(V.hyp.unique())}
V3=pd.read_csv('3e174503-V3_first_snippet_group.csv'); m3=V3[V3.method=='mean_flip']
g=m3.pivot_table(index='measure',columns=['condition','group'],values='p_fdr'); med=m3.pivot_table(index='measure',columns=['condition','group'],values='median_first_minus_late')
rows=[]
for mm in order:
    r=[mm]
    for c in ['baseline','sad','fear']:
        sig=g.loc[mm,c]<.05; n=int(sig.sum()); sgn=np.sign(med.loc[mm,c][sig]).mean() if n else 0
        r.append(f"{n}/4 {'higher' if sgn>0 else ('lower' if sgn<0 else '')}".strip())
    rows.append(r)
D['first']=rows; D['first_counts']={c:int((m3[m3.condition==c].p_fdr<.05).sum()) for c in ['baseline','sad','fear']}
NV=pd.read_csv('4f19dcc5-P2_network_specific_variance.csv'); nv=NV.pivot_table(index='measure',columns='method',values='network_specific_share')
D['netvar']=[[mm,f(nv.loc[mm,'mean_flip']*100,1),f(nv.loc[mm,'pca_flip']*100,1)] for mm in order]
NS=pd.read_csv('23e6802f-P2_network_deviation_stability.csv'); ns=NS[NS.method=='mean_flip'].groupby(['network','factor']).rho.agg(['min','max','median']).reset_index()
D['netstab']=[[r['network'],r['factor'],f(r['min']),f(r['max']),f(r['median'])] for _,r in ns.iterrows()]
ND=pd.read_csv('b315ee9c-P2_network_deviation_vs_behaviour.csv'); nd=ND[ND.p<.05].sort_values('p')
D['nethits']=[[r.method,r.condition,r.network,r.factor,names[r.behaviour],f(r.rho),pf(r.p)] for r in nd.itertuples()]
D['nethit_count']=[int((ND.p<.05).sum()),int(ND.p.notna().sum())]
PS=pd.read_csv('03ae3bee-P3_profile_selection.csv'); D['profiles']=[[r.method,int(r.k),f(r.bic,1),int(r.min_class_n),f(r.stability_ari),'yes' if r.passes else 'no'] for r in PS.itertuples()]
HS=pd.read_csv('2d76a8b5-P4_hmm_selection.csv'); D['hmmsel']=[[r.method,int(r.K),f(r.bic,0),f(r.cv_loglik_per_snippet,3),f(r.min_state_occupancy),'yes' if r.selected else ''] for r in HS.itertuples()]
HB=pd.read_csv('af9ffc2c-P4_hmm_subject_block.csv'); D['hmmsw']={mt:[f((gg.switch_rate>0).mean()*100,0),f(gg.mean_dwell_snippets.median(),1)] for mt,gg in HB.groupby('method')}
HT=pd.read_csv('53b7af93-P4_hmm_transmat.csv'); sd=HT[HT.from_state==HT.to_state]; D['hmmstay']=[f(sd.p.min()),f(sd.p.max())]
HV=pd.read_csv('dc60d5dd-P4_hmm_vs_behaviour.csv'); LK=pd.read_csv('f97153df-P4_state_locked_ratings.csv')
D['hmmbeh']=[int((HV.p<.05).sum()),int(HV.p.notna().sum()),int((LK.p_wilcoxon<.05).sum()),len(LK)]
Cf=pd.read_csv('5f6a535f-C0_confounds.csv',dtype={'subj':str}); Cf.loc[Cf.subj=='037','heart_rate_bpm']=np.nan
BB=pd.read_csv('6a1ba10c-V0_behaviour_table.csv',dtype={'subj':str}); Bm=BB.groupby(['subj','condition'])[['latent_sd','mean','sd']].mean().reset_index()
conf=['head_x_mm','head_y_mm','head_z_mm','head_dist_mm','meg_mag_rms_fT','meg_grad_rms','heart_rate_bpm','blink_rate_per_min','ica_n_components']
cn={'head_x_mm':'Head position x','head_y_mm':'Head position y','head_z_mm':'Head position z','head_dist_mm':'Head–helmet distance','meg_mag_rms_fT':'Signal size, magnetometers (fT)','meg_grad_rms':'Signal size, gradiometers','heart_rate_bpm':'Heart rate (bpm)','blink_rate_per_min':'Blink rate (/min)','ica_n_components':'ICA components fitted'}
def part(x,y,Z):
    msk=np.isfinite(x)&np.isfinite(y)&np.isfinite(Z).all(1); rx,ry=stats.rankdata(x[msk]),stats.rankdata(y[msk])
    R=np.column_stack([np.ones(msk.sum())]+[stats.rankdata(z) for z in Z[msk].T])
    ex=rx-R@np.linalg.lstsq(R,rx,rcond=None)[0]; ey=ry-R@np.linalg.lstsq(R,ry,rcond=None)[0]
    r=np.corrcoef(ex,ey)[0,1]; df=msk.sum()-2-Z.shape[1]; t=r*np.sqrt(df/(1-r*r)); return r,2*stats.t.sf(abs(t),df),int(msk.sum())
cr=[];adj=[]
for meth in ['mean_flip','pca_flip']:
    W=F[F.method==meth].merge(Cf,on=['subj','condition']).merge(Bm,on=['subj','condition'],how='left')
    b=W[W.condition=='baseline']
    if meth=='mean_flip':
        for c in conf:
            r1=stats.spearmanr(b.F1,b[c],nan_policy='omit'); r2=stats.spearmanr(b.F2,b[c],nan_policy='omit')
            cr.append([cn[c],f(r1[0]),pf(r1[1]),f(r2[0]),pf(r2[1])])
    for cond in ['baseline','sad','fear']:
        g=W[W.condition==cond]
        r0=part(g.F1.values,g.latent_sd.values,g[['mean']].values); r1=part(g.F1.values,g.latent_sd.values,g[['mean']+conf].values)
        adj.append([meth,cond,r0[2],f(r0[0]),pf(r0[1]),r1[2],f(r1[0]),pf(r1[1])])
D['confcorr']=cr; D['confadj']=adj
desc=Cf[conf].describe()
D['confdesc']=[[cn[c],f(desc.loc['50%',c],1),f(desc.loc['min',c],1),f(desc.loc['max',c],1),int(desc.loc['count',c])] for c in conf]
M1=pd.read_csv('04803ce9-M1_between_subject.csv'); pr=M1[M1.primary]
D['m1']=[[r.behav_condition,r.emotion,r.method,int(r.n_subjects),f(r.pearson_r),pf(r.p_pearson)] for r in pr.itertuples()]
D['m1_counts']=[int((M1.p_pearson<.05).sum()),len(M1),f(M1.p_pearson_fdr.min(),3)]
M2=pd.read_csv('861501a2-M2_within_subject.csv'); D['m2_counts']=[int((M2.p_meg<.05).sum()),len(M2),f(M2.p_meg_fdr.min(),2)]
D['m2_top']=[[r.emotion,r.method,r.group,r.measure,f(r.beta_meg),pf(r.p_meg),f(r.p_meg_fdr)] for r in M2.sort_values('p_meg').head(6).itertuples()]
M3=pd.read_csv('df36aba3-M3_decoding.csv'); D['m3']=[[r.method,r.emotion,int(r.n_obs),int(r.n_subjects),f(r.r_out_of_sample,3),f(r.null_p95,3),pf(r.p_perm)] for r in M3.itertuples()]
json.dump(D,open(OUT,'w'),ensure_ascii=False,indent=0)
print('ok', D['confadj'], D['stability'], D['m1'], D['m2_counts'], D['hmmstay'], D['fb_count'], D['first_counts'], sep='\n')
