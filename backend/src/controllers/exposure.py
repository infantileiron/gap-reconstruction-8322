from src.services import exposure as service


def list_items(run_id):
    return service.list_reviews(run_id)


def create(run_id, data, user):
    review, cached = service.review_run(run_id, data, user)
    return review
