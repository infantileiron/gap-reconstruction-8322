# 测试用最小桩：仅满足 backend 模块的顶层 import 与路由结构检查，不运行真实服务
class HTTPException(Exception):
    def __init__(self, status_code=500, detail=""):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
class _Route:
    def __init__(self, path, methods):
        self.path = path; self.methods = methods
class APIRouter:
    def __init__(self, prefix="", tags=None, **kw):
        self.prefix = prefix; self.routes = []
    def _add(self, path, methods):
        def deco(fn):
            self.routes.append(_Route(self.prefix + path, methods)); return fn
        return deco
    def get(self, path, **kw): return self._add(path, {"GET"})
    def post(self, path, **kw): return self._add(path, {"POST"})
    def patch(self, path, **kw): return self._add(path, {"PATCH"})
    def delete(self, path, **kw): return self._add(path, {"DELETE"})
def Depends(fn=None, **kw): return fn
class Request: pass
class Response:
    def __init__(self, content="", media_type=None, headers=None): pass
class FastAPI:
    def __init__(self, *a, **k):
        self.routes = []
    def include_router(self, router):
        self.routes.extend(router.routes)
    def add_middleware(self, *a, **k):
        pass
    def get(self, path, **kw):
        def deco(fn):
            self.routes.append(_Route(path, {"GET"})); return fn
        return deco
def asynccontextmanager(fn):
    class _CM:
        def __call__(self, app): return self
        def __enter__(self): return None
        def __exit__(self, *a): return False
    return _CM()
