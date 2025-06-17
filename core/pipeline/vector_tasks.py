from core.celery_app import celery_app
from core.pipeline.vector_pipeline import VectorPipeline

def _get_pipeline():
    return VectorPipeline()

@celery_app.task(name='core.pipeline.vector_tasks.process_image')
def process_image(image_path):
    pipeline = _get_pipeline()
    # 실제 구현에서는 image_path로부터 이미지를 로드해야 함
    # 예시: image = load_image(image_path)
    image = None  # TODO: 실제 이미지 로딩 코드로 대체
    result = pipeline.run(image)
    return result 