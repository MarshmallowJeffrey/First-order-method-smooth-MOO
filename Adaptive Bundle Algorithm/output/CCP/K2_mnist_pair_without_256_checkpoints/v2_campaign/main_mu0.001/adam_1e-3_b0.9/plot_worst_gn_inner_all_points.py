from pathlib import Path
import json, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path('/Users/shirch/vscode101/.venv/First-order-method-smooth-MOO/Adaptive Bundle Algorithm/output/CCP/K2_mnist_pair_without_256_checkpoints/v2_campaign/main_mu0.001/adam_1e-3_b0.9')
R=json.loads((P/'inner_range_gn_K2.json').read_text())['runs']
A=R['adaptive_ccp_seed41']
rows=[]
for name,r in R.items():
 if r['family']=='adaptive':continue
 y=np.asarray(r['inner_norm']); i=int(np.argmin(y))
 rows.append(dict(run=name,family=r['family'],param=int(name.split('_')[1][1:]),gn=float(y[i]),budget=float(r['ck_grads'][i]),time=float(r['ck_cpu'][i]),checkpoint=i))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.22,'grid.linewidth':.6,'savefig.dpi':220})
U={'budget':{10:(8,7),20:(8,7),30:(8,7),40:(-8,17),50:(9,7),70:(10,0),90:(-7,10),120:(9,-6),140:(-9,-20)},'time':{10:(8,7),20:(8,7),30:(8,7),40:(9,5),50:(-3,17),70:(10,0),90:(-8,10),120:(8,-6),140:(-9,-21)}}
S={'budget':{10:(8,7),20:(-8,-7),30:(0,-17),40:(-8,12),50:(8,7),70:(1,20),90:(8,7),120:(-8,12),140:(-8,12)},'time':{10:(8,7),20:(-8,-10),30:(2,-17),40:(-9,13),50:(12,5),70:(1,23),90:(8,7),120:(-8,12),140:(-8,12)}}
for tag,key,label,lim in [('budget','ck_grads','Gradient evaluations',21700),('time','ck_cpu','Wall-clock seconds',850)]:
 fig,ax=plt.subplots(figsize=(12,5.8))
 ax.plot(A[key],np.minimum.accumulate(A['inner_norm']),lw=2.6,color='#ff7f0e',label='Adaptive CCP',zorder=2)
 for fam,col,m,offsets,labeltext in [('uniform','#377eb8','s',U,'Uniform grid (r)'),('surf','#e41a1c','^',S,'SURF (N)')]:
  rr=[r for r in rows if r['family']==fam]
  ax.scatter([r[tag] for r in rr],[r['gn'] for r in rr],s=48,color=col,marker=m,edgecolor='white',linewidth=.6,label=labeltext,zorder=4)
  for r in rr:
   dx,dy=offsets[tag][r['param']]
   ax.annotate(str(r['param']),(r[tag],r['gn']),xytext=(dx,dy),textcoords='offset points',ha='right' if dx<0 else 'left' if dx>0 else 'center',va='center',color=col,fontsize=10.5,zorder=5)
 ax.set(yscale='log',ylim=(.0035,14),xlim=(-600 if tag=='budget' else -22,lim),xlabel=label,ylabel='Best-so-far worst-case gradient norm')
 ax.grid(False,which='minor');ax.legend(loc='upper right',fontsize=11)
 fig.tight_layout(rect=(0,.055,1,1),pad=1)
 fig.text(.5,.025,'All 9 uniform configurations and 9 SURF configurations are shown; no points omitted.',ha='center',fontsize=10,color='#555555')
 for ext in ['png','svg']:
  out=P/f'worst_gn_dots_inner_all_points_{tag}.{ext}';fig.savefig(out)
  if ext=='png':shutil.copy2(out,Path('/Users/shirch/Desktop')/f'K=2_MNIST_worst_gn_dots_inner_all_points_{tag}.png')
 plt.close(fig)
(P/'worst_gn_dots_inner_all_points_metadata.json').write_text(json.dumps({'source':'inner_range_gn_K2.json','selection':'Each baseline: minimum stored inner_norm over checkpoints; x is first checkpoint attaining that minimum. Adaptive: cumulative minimum. No coordinate jitter, omitted points or smoothing.','rows':rows},indent=2))
shutil.copy2(__file__,P/'plot_worst_gn_inner_all_points.py')
print(P)
