import {useState} from 'react';
import * as exposureApi from '../api/exposure';
import {useResource} from '../hooks/useResource';
import {refreshResources} from '../stores/resources';
import {decimal} from '../utils/format';
import type {ExposureReview, Run} from '../types/entities';

const f4=(v:number|null|undefined)=>v==null?'—':decimal(v,4);

export default function ExposureReviewPanel({run}:{run:Run}){
 const {data,error}=useResource(()=>exposureApi.listByRun(run.id),[]);
 const [boundaryText,setBoundaryText]=useState('0, 120, 240, 479');
 const [mode,setMode]=useState<'production'|'validation'>('production');
 const [message,setMessage]=useState('');
 const [busy,setBusy]=useState(false);
 const reviews:ExposureReview[]=data||[];

 async function submit(e:React.FormEvent){
  e.preventDefault();
  const periods=boundaryText.split(/[,，\s]+/).map(s=>s.trim()).filter(Boolean).map(Number);
  if(periods.some(x=>!Number.isFinite(x))||periods.length<2){setMessage('请输入至少两个数值边界，用逗号或空格分隔');return;}
  setBusy(true);setMessage('');
  try{
   await exposureApi.create(run.id,{periods,mode});
   refreshResources();
   setMessage('');
  }catch(err:any){setMessage(err.message);}
  finally{setBusy(false);}
 }

 return <section className="paper">
  <h2>时段累计暴露量复核（沿重建曲线分段线性积分）</h2>
  <p style={{fontSize:12,maxWidth:1000}}>
   按实际时间对重建曲线作梯形积分，边界可落在观测之间（确定性线性插值，不改变总量与方差）。
   实测值原样保留，仅原始缺口的补值进入不确定度并保留跨时段相关性；遮挡复核用独立训练数组重算。
   单次计算额外内存上限64MB，超限拒绝且保留已有复核。
  </p>
  {error&&<p className="notice">{error}</p>}
  {message&&<p className="notice">{message}</p>}
  <form className="form-grid" onSubmit={submit}>
   <label className="wide">时段边界（小时，逗号/空格分隔，至少2个）
    <input value={boundaryText} onChange={e=>setBoundaryText(e.target.value)} aria-label="时段边界"/>
   </label>
   <label>复核口径
    <select value={mode} onChange={e=>setMode(e.target.value as 'production'|'validation')} aria-label="复核口径">
     <option value="production">生产重建（全部原始观测）</option>
     <option value="validation">遮挡复核（目标置空，独立训练）</option>
    </select>
   </label>
   <label style={{alignSelf:'end'}}><button className="primary" disabled={busy}>{busy?'计算中…':'发起时段累计复核'}</button></label>
  </form>

  {reviews.length===0
   ? <div className="empty" style={{marginTop:20}}>尚无时段累计复核，输入边界后发起。</div>
   : reviews.map(er=><ExposureCard key={er.id} er={er}/>)}
 </section>;
}

function ExposureCard({er}:{er:ExposureReview}){
 const t=er.total, rc=er.reconciliation;
 return <article className="paper" style={{borderTopColor:'oklch(55% 0.09 65)'}}>
  <div className="toolbar" style={{marginBottom:12}}>
   <div><b>{er.mode==='production'?'生产重建累计复核':'遮挡独立累计复核'}</b>
    <small>{er.id} · {er.station} · 时段数 {er.period_count} · 网格点 {er.grid_point_count} · 查询 {er.query_count} 次</small>
    <small>额外内存峰值 {(er.memory_peak_bytes/1024/1024).toFixed(2)}MB / 上限 {(er.memory_limit_bytes/1024/1024).toFixed(0)}MB · {er.created_at}</small>
   </div>
  </div>

  <div className="method-comparison">
   <div><b>整段总量</b><p>{f4(t.total_mean)}（{decimal(t.start_hour,2)}–{decimal(t.end_hour,2)}h，{decimal(t.duration_hour,2)}h）</p></div>
   <div><b>实测 / 补值贡献</b><p>{f4(t.observed_contribution)} / {f4(t.imputed_contribution)}</p></div>
   <div><b>整体95%区间（保留补值相关性）</b><p>[{f4(t.total_low)}, {f4(t.total_high)}] · SD {f4(t.sd)}</p></div>
  </div>

  <div className="table-scroll">
   <table>
    <thead><tr><th>时段</th><th>起(h)</th><th>止(h)</th><th>时长(h)</th><th>总量</th><th>实测贡献</th><th>补值贡献</th><th>方差</th><th>SD</th><th>95%下限</th><th>95%上限</th></tr></thead>
    <tbody>{er.periods_detail.map(p=><tr key={p.index}>
     <td>{p.index}</td><td>{decimal(p.start_hour,3)}</td><td>{decimal(p.end_hour,3)}</td><td>{decimal(p.duration_hour,3)}</td>
     <td>{f4(p.total_mean)}</td><td>{f4(p.observed_contribution)}</td><td>{f4(p.imputed_contribution)}</td>
     <td>{f4(p.variance)}</td><td>{f4(p.sd)}</td><td>{f4(p.total_low)}</td><td>{f4(p.total_high)}</td>
    </tr>)}</tbody>
   </table>
  </div>

  <h3 style={{fontSize:14,margin:'18px 0 8px'}}>相邻时段协方差对账</h3>
  <div className="table-scroll">
   <table>
    <thead><tr><th>左时段</th><th>右时段</th><th>协方差</th><th>相关系数</th></tr></thead>
    <tbody>{er.adjacent_covariance.length===0
     ? <tr><td colSpan={4}>仅一个时段，无相邻对账项</td></tr>
     : er.adjacent_covariance.map(a=><tr key={a.left_index}>
      <td>{a.left_index}</td><td>{a.right_index}</td><td>{f4(a.covariance)}</td>
      <td>{a.correlation==null?'—':decimal(a.correlation,4)}</td></tr>)}</tbody>
   </table>
  </div>

  <h3 style={{fontSize:14,margin:'18px 0 8px'}}>整段还原对账</h3>
  <div className="table-scroll">
   <table>
    <tbody>
     <tr><td>分段总量和 − 整段总量（均值残差）</td><td>{f4(rc.mean_residual)}</td></tr>
     <tr><td>协方差矩阵全求和 − 首末积分状态方差（方差残差）</td><td>{f4(rc.variance_residual)}</td></tr>
     <tr><td>实测+补值 − 总量（拆分残差）</td><td>{f4(rc.observed_plus_imputed_residual)}</td></tr>
     <tr><td>仅相邻协方差口径残差（非相邻时段亦相关，预期非0）</td><td>{f4(rc.adjacent_only_variance_residual)}</td></tr>
    </tbody>
   </table>
  </div>
 </article>;
}
