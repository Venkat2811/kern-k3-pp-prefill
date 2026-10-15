import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('reference',type=Path);p.add_argument('candidate',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--label',default='candidate-vs-reference');a=p.parse_args()
afiles=sorted(a.reference.glob('*.json'));bfiles=sorted(a.candidate.glob('*.json'))
assert len(afiles)==len(bfiles)==3,'All three author shapes required'
assert [x.name for x in afiles]==[x.name for x in bfiles]
rows=[]
for x,y in zip(afiles,bfiles):
 ja,jb=json.loads(x.read_text()),json.loads(y.read_text())
 assert ja==jb,(x.name,'state/hidden image differs; stop performance promotion')
 assert len(ja['images'])==13,('expected hidden +12 layer states',x.name)
 assert all(z['nonfinite']==0 for z in ja['images'])
 rows.append({'case':x.name,'bitwise_images':len(ja['images']),'all_finite':True})
a.out.write_text(json.dumps({'passed':True,'comparison':a.label,'scope':'full hidden + all12 KV/KDA state image SHA256 equality on same real-history trajectory; gate52 only','cases':rows},indent=2)+'\n')
print(json.dumps(rows))
