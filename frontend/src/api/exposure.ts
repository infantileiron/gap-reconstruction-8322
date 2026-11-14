import {request} from './client';
import type {ExposureReview} from '../types/entities';

// 时段累计暴露量复核：挂在运行下，超限返回413由 client 透传 detail
export const listByRun=(runId:string)=>request<ExposureReview[]>(`/runs/${runId}/exposure-reviews`);
export const create=(runId:string,data:{periods:number[];mode:'production'|'validation'})=>
 request<ExposureReview>(`/runs/${runId}/exposure-reviews`,'POST',data);
