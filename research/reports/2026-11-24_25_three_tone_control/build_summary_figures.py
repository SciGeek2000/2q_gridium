"""Publication summaries from packaged saved results only; no simulation imports."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
P=Path(__file__).resolve().parent
m=json.loads((P/'evidence/3OAYR1_numerical_comparison.json').read_text())['metrics']
keys=['baseline56','high_only56','middle_only56','low_only56','compensated56','low_compensated56']
labels=['Baseline','High +1 MHz','Middle +1 MHz','Low −1 MHz','Middle/high +1 MHz','Low −1, middle +1 MHz']
colors=['#404040','#b34735','#b34735','#b34735','#007d8a','#007d8a']
fig,ax=plt.subplots(figsize=(10,5),layout='constrained');y=np.arange(6)
v=[m[k]['lab_X'] for k in keys];ax.barh(y,v,color=colors);ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,1.1);ax.set_xlabel('Leakage-inclusive lab-frame X fidelity');ax.set_title('Physical carrier-frequency sensitivity • 56 levels\nNew summary from saved numerical evidence; unchanged pulse otherwise')
for i,f in enumerate(v):ax.text(f+.012,i,f'{f:.6f}',va='center')
ax.grid(axis='x',alpha=.2);fig.savefig(P/'figures/02_physical_frequency_fidelity.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
for off,key,label,col in [(-.24,'input0','Input 0','#0077bb'),(0,'input1','Input 1','#ee7733'),(.24,'worst','Worst superposition','#555555')]:
 values=[100*(m[k]['worst_leakage'] if key=='worst' else m[k]['basis_leakage'][0 if key=='input0' else 1]) for k in keys]
 ax.barh(y+off,values,height=.22,label=label,color=col)
ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,85);ax.set_xlabel('Endpoint leakage (%)');ax.set_title('Physical endpoint leakage • 56 levels\nNew summary from saved numerical evidence');ax.legend(loc='lower right');ax.grid(axis='x',alpha=.2)
fig.savefig(P/'figures/03_physical_endpoint_leakage.png',dpi=180);plt.close(fig)
