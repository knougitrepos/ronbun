import os

from DB.db_utils import SessionLocal
from core.database import VectorRepository
from core.pipeline.vector_tasks import process_image
from core.schemas import Embedding256, EmbeddingPQ


DATASET_ROOT = "downloaded_datasets"


def get_image_paths(root):
    image_paths = []
    for dirpath, _, files in os.walk(root):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_paths.append(os.path.join(dirpath, file))
    return image_paths


def main():
    session = SessionLocal()
    repository = VectorRepository(session)
    queued = 0

    for image_path in get_image_paths(DATASET_ROOT):
        image = repository.get_image_by_path(image_path)
        if image is None:
            process_image.delay(image_path)
            queued += 1
            continue

        origin_exists = repository.get_embeddings_512(
            image_id=image.id,
            vector_type="origin",
            param_filter={},
        )
        if not origin_exists:
            process_image.delay(image_path)
            queued += 1

    pca_count = session.query(Embedding256).filter(Embedding256.vector_type == "pca").count()
    pq_count = session.query(EmbeddingPQ).filter(EmbeddingPQ.vector_type == "pq").count()
    session.close()

    print(f"Queued origin extraction jobs: {queued}")
    print(f"Existing PCA embeddings: {pca_count}")
    print(f"Existing PQ codes: {pq_count}")


if __name__ == "__main__":
    main()
