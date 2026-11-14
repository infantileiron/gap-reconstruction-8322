from fastapi import APIRouter, Depends

from src.controllers import exposure as controller
from src.schemas.entities import ExposureReviewInput
from src.services.auth import current, editor

router = APIRouter(prefix="/runs", tags=["exposure-reviews"])


@router.get("/{run_id}/exposure-reviews")
def list_items(run_id: str, user=Depends(current)):
    return controller.list_items(run_id)


@router.post("/{run_id}/exposure-reviews", status_code=201)
def create(run_id: str, data: ExposureReviewInput, user=Depends(editor)):
    # 内存超64MB由服务层抛413；重复时段查询命中缓存，不重复计算
    return controller.create(run_id, data, user)
