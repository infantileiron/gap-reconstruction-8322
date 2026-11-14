import csv,io
from fastapi import HTTPException
from src.models.entities import Report
from src.config.settings import ARTIFACT_DIR
from src.utils import repository as repo
from src.utils.identity import new_id,now

def safe_cell(value):return "'"+value if isinstance(value,str) and value.lstrip().startswith(('=','+','-','@')) else value
def list_reports():return repo.all_rows('reports')
def create(data):
 run,review=repo.get('runs',data.run_id),repo.get('reviews',data.review_id)
 if review['status']!='approved' or review['run_id']!=run['id'] or run['status']!='completed':raise ValueError('报告必须引用该运行批准复核')
 report=repo.put('reports',Report(new_id('rpt'),data.run_id,data.review_id,data.title,now()));ARTIFACT_DIR.mkdir(parents=True,exist_ok=True);(ARTIFACT_DIR/(report['id']+'.csv')).write_text(download(report['id']),encoding='utf-8');return report
def download(id):
 report=repo.get('reports',id);run=repo.get('runs',report['run_id']);r=run['result'];review=repo.get('reviews',report['review_id'])
 if review['status']!='approved':raise HTTPException(403,'复核未通过')
 out=io.StringIO();w=csv.writer(out);m=r['validation']['metrics']
 w.writerow(['report','station','points','missing','pchip_unfilled_gaps','q_per_hour','observation_r','validation_targets','common_count','rts_mae_common','pchip_mae_common','rts_observation_interval_coverage','checksum']);w.writerow([safe_cell(report['title']),safe_cell(r['station']),r['point_count'],r['missing_count'],r['pchip_unfilled_gaps'],r['study']['q_per_hour'],r['study']['observation_r'],m['target_count'],m['common_count'],m['rts_mae_common'],m['pchip_mae_common'],m['rts_observation_interval_coverage'],r['checksum']])
 w.writerow([]);fields=['time_hour','original_value','is_gap','rts_output','pchip_output','rts_mean','state_variance','observation_variance','state_low','state_high','observation_low','observation_high'];w.writerow(fields)
 for p in r['production']['series']:w.writerow([p[k] for k in fields])
 w.writerow([]);fields=['time_hour','validation_truth','training_value','rts_mean','pchip','state_variance','observation_variance','observation_low','observation_high'];w.writerow(fields)
 for p in r['validation']['targets']:w.writerow([p[k] for k in fields])
 _write_exposure_sections(w,r)
 return '\ufeff'+out.getvalue()

def _write_exposure_sections(w,r):
 """\u65f6\u6bb5\u7d2f\u8ba1\u66b4\u9732\u91cf\u590d\u6838\u660e\u7ec6\uff1a\u4e0e\u9875\u9762 /exposure-reviews \u67e5\u8be2\u540c\u6e90\uff0c\u968f\u62a5\u544a\u5f52\u6863\u3002"""
 reviews=r.get('exposure_reviews',[])
 for er in reviews:
  w.writerow([]);w.writerow(['exposure_review','review_id','mode','station','created_at','periods','grid_point_count','memory_peak_kb','memory_limit_kb'])
  w.writerow([er['id'],er['id'],er['mode'],safe_cell(er['station']),er['created_at'],';'.join(str(x) for x in er['periods']),er['grid_point_count'],round(er['memory_peak_bytes']/1024,3),round(er['memory_limit_bytes']/1024,3)])
  t=er['total']
  w.writerow(['total','start_hour','end_hour','duration_hour','total_mean','observed_contribution','imputed_contribution','variance','sd','total_low','total_high'])
  w.writerow(['total',t['start_hour'],t['end_hour'],t['duration_hour'],t['total_mean'],t['observed_contribution'],t['imputed_contribution'],t['variance'],t['sd'],t['total_low'],t['total_high']])
  w.writerow(['period','start_hour','end_hour','duration_hour','total_mean','observed_contribution','imputed_contribution','variance','sd','total_low','total_high'])
  for p in er['periods_detail']:
   w.writerow([p['index'],p['start_hour'],p['end_hour'],p['duration_hour'],p['total_mean'],p['observed_contribution'],p['imputed_contribution'],p['variance'],p['sd'],p['total_low'],p['total_high']])
  w.writerow(['adjacent_covariance','left_index','right_index','covariance','correlation'])
  for a in er['adjacent_covariance']:
   w.writerow(['adjacent',a['left_index'],a['right_index'],a['covariance'],a['correlation']])
  rc=er['reconciliation']
  w.writerow(['reconciliation','sum_period_mean','direct_total_mean','mean_residual','sum_period_variance','direct_total_variance','variance_residual','adjacent_covariance_sum','adjacent_only_variance_residual','observed_plus_imputed_residual'])
  w.writerow(['reconciliation',rc['sum_period_mean'],rc['direct_total_mean'],rc['mean_residual'],rc['sum_period_variance'],rc['direct_total_variance'],rc['variance_residual'],rc['adjacent_covariance_sum'],rc['adjacent_only_variance_residual'],rc['observed_plus_imputed_residual']])
