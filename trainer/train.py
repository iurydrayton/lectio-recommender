"""
train.py — Two-tower model com TensorFlow para recomendação de livros.

Dependências:
    pip install tensorflow sqlalchemy psycopg2-binary numpy

Uso:
    DATABASE_URL=postgresql://user:pass@localhost:5432/books_db python train.py
    EMBEDDINGS_DIR=/app/embeddings python train.py  # dentro do Docker
"""

import os
import random
import numpy as np
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

DATABASE_URL   = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/books_db")
EMBEDDINGS_DIR = Path(os.getenv("EMBEDDINGS_DIR", "./embeddings"))

EMBEDDING_DIM = 64
BATCH_SIZE    = 256
EPOCHS        = 20
LEARNING_RATE = 1e-3
N_WORKERS     = 3
NEG_RATIO     = 3
RANDOM_SEED   = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Leitura paralela do PostgreSQL
# ---------------------------------------------------------------------------

def fetch_partition(partition: int, n_partitions: int, db_url: str) -> list[tuple[int, int]]:
    engine = create_engine(db_url, echo=False)
    query  = text("""
        SELECT user_id, book_id
        FROM purchases
        WHERE MOD(id, :n) = :part
    """)
    with Session(engine) as session:
        rows = session.execute(query, {"n": n_partitions, "part": partition}).fetchall()
    engine.dispose()
    return [(r.user_id, r.book_id) for r in rows]


def fetch_all_ids(db_url: str) -> tuple[list[int], list[int]]:
    engine = create_engine(db_url, echo=False)
    with Session(engine) as session:
        user_ids = [r[0] for r in session.execute(text("SELECT id FROM users ORDER BY id")).fetchall()]
        book_ids = [r[0] for r in session.execute(text("SELECT id FROM books ORDER BY id")).fetchall()]
    engine.dispose()
    return user_ids, book_ids


def load_purchases_parallel(db_url: str, n_workers: int) -> list[tuple[int, int]]:
    print(f"📂 Lendo compras em {n_workers} workers paralelos...")
    purchases = []

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(fetch_partition, i, n_workers, db_url): i
            for i in range(n_workers)
        }
        for future in as_completed(futures):
            part = futures[future]
            rows = future.result()
            print(f"   Worker {part + 1}: {len(rows)} compras carregadas")
            purchases.extend(rows)

    print(f"   Total: {len(purchases)} compras\n")
    return purchases


# ---------------------------------------------------------------------------
# Preparação dos dados
# ---------------------------------------------------------------------------

def build_index_maps(user_ids: list[int], book_ids: list[int]) -> tuple[dict, dict]:
    user_to_idx = {uid: i for i, uid in enumerate(sorted(user_ids))}
    book_to_idx = {bid: i for i, bid in enumerate(sorted(book_ids))}
    return user_to_idx, book_to_idx


def build_dataset(
    purchases: list[tuple[int, int]],
    user_to_idx: dict,
    book_to_idx: dict,
    all_book_idxs: list[int],
    neg_ratio: int = NEG_RATIO,
) -> tf.data.Dataset:
    positive_set = set(purchases)
    user_idxs, book_idxs, labels = [], [], []

    for (uid, bid) in purchases:
        u_idx = user_to_idx[uid]
        b_idx = book_to_idx[bid]

        user_idxs.append(u_idx)
        book_idxs.append(b_idx)
        labels.append(1.0)

        neg_count = 0
        attempts  = 0
        while neg_count < neg_ratio and attempts < neg_ratio * 10:
            attempts += 1
            neg_bid = random.choice(all_book_idxs)
            if (uid, list(book_to_idx.keys())[neg_bid]) not in positive_set:
                user_idxs.append(u_idx)
                book_idxs.append(neg_bid)
                labels.append(0.0)
                neg_count += 1

    user_idxs = np.array(user_idxs, dtype=np.int32)
    book_idxs = np.array(book_idxs, dtype=np.int32)
    labels    = np.array(labels,    dtype=np.float32)

    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    user_idxs, book_idxs, labels = user_idxs[idx], book_idxs[idx], labels[idx]

    dataset = tf.data.Dataset.from_tensor_slices(
        ({"user_idx": user_idxs, "book_idx": book_idxs}, labels)
    )
    return dataset.shuffle(4096).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Modelo Two-Tower
# ---------------------------------------------------------------------------

def build_two_tower_model(n_users: int, n_books: int, embedding_dim: int) -> tf.keras.Model:
    user_input = tf.keras.Input(shape=(), dtype=tf.int32, name="user_idx")
    book_input = tf.keras.Input(shape=(), dtype=tf.int32, name="book_idx")

    # Torre do usuário
    user_emb = tf.keras.layers.Embedding(
        input_dim=n_users,
        output_dim=embedding_dim,
        embeddings_regularizer=tf.keras.regularizers.l2(1e-4),
        name="user_embedding",
    )(user_input)
    user_emb = tf.keras.layers.Dense(128, activation="relu", name="user_dense_1")(user_emb)
    user_emb = tf.keras.layers.Dropout(0.2)(user_emb)
    user_emb = tf.keras.layers.Dense(embedding_dim, name="user_dense_2")(user_emb)
    user_emb = tf.keras.layers.Lambda(
        lambda x: tf.math.l2_normalize(x, axis=-1), name="user_l2_norm"
    )(user_emb)

    # Torre do livro
    book_emb = tf.keras.layers.Embedding(
        input_dim=n_books,
        output_dim=embedding_dim,
        embeddings_regularizer=tf.keras.regularizers.l2(1e-4),
        name="book_embedding",
    )(book_input)
    book_emb = tf.keras.layers.Dense(128, activation="relu", name="book_dense_1")(book_emb)
    book_emb = tf.keras.layers.Dropout(0.2)(book_emb)
    book_emb = tf.keras.layers.Dense(embedding_dim, name="book_dense_2")(book_emb)
    book_emb = tf.keras.layers.Lambda(
        lambda x: tf.math.l2_normalize(x, axis=-1), name="book_l2_norm"
    )(book_emb)

    score  = tf.keras.layers.Dot(axes=-1, name="dot_product")([user_emb, book_emb])
    output = tf.keras.layers.Activation("sigmoid", name="output")(score)

    return tf.keras.Model(
        inputs=[user_input, book_input],
        outputs=output,
        name="two_tower_recommender",
    )


# ---------------------------------------------------------------------------
# Extração dos embeddings
# ---------------------------------------------------------------------------

def extract_embeddings(
    model: tf.keras.Model,
    user_to_idx: dict,
    book_to_idx: dict,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    user_model = tf.keras.Model(
        inputs=model.get_layer("user_embedding").input,
        outputs=model.get_layer("user_l2_norm").output,
    )
    book_model = tf.keras.Model(
        inputs=model.get_layer("book_embedding").input,
        outputs=model.get_layer("book_l2_norm").output,
    )

    user_idxs   = np.array(list(user_to_idx.values()), dtype=np.int32)
    book_idxs   = np.array(list(book_to_idx.values()), dtype=np.int32)

    user_vectors = user_model.predict(user_idxs, verbose=0)
    book_vectors = book_model.predict(book_idxs, verbose=0)

    user_embeddings = {uid: user_vectors[idx] for uid, idx in user_to_idx.items()}
    book_embeddings = {bid: book_vectors[idx] for bid, idx in book_to_idx.items()}

    return user_embeddings, book_embeddings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Carrega dados
    user_ids, book_ids = fetch_all_ids(DATABASE_URL)
    purchases          = load_purchases_parallel(DATABASE_URL, N_WORKERS)

    print(f"👥 Usuários: {len(user_ids)} | 📚 Livros: {len(book_ids)} | 🛒 Compras: {len(purchases)}")

    # 2. Index maps
    user_to_idx, book_to_idx = build_index_maps(user_ids, book_ids)
    all_book_idxs = list(book_to_idx.values())

    # 3. Dataset
    print("\n⚙️  Construindo dataset com negative sampling...")
    dataset = build_dataset(purchases, user_to_idx, book_to_idx, all_book_idxs)
    print(f"   {len(dataset)} batches de {BATCH_SIZE} amostras\n")

    # 4. Modelo
    model = build_two_tower_model(len(user_ids), len(book_ids), EMBEDDING_DIM)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    model.summary()

    # 5. Treino
    print("\n🚀 Iniciando treinamento...\n")
    model.fit(
        dataset,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="loss", patience=3, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="loss", factor=0.5, patience=2, min_lr=1e-5
            ),
        ],
    )

    # 6. Salva modelo
    model_path = EMBEDDINGS_DIR / "two_tower_model.keras"
    model.save(model_path)
    print(f"\n💾 Modelo salvo em {model_path}")

    # 7. Extrai e salva embeddings
    print("\n🔢 Extraindo embeddings...")
    user_embeddings, book_embeddings = extract_embeddings(model, user_to_idx, book_to_idx)

    np.save(EMBEDDINGS_DIR / "user_embeddings.npy", user_embeddings)
    np.save(EMBEDDINGS_DIR / "book_embeddings.npy", book_embeddings)

    print(f"   ✅ user_embeddings.npy — {len(user_embeddings)} vetores [{EMBEDDING_DIM}d]")
    print(f"   ✅ book_embeddings.npy — {len(book_embeddings)} vetores [{EMBEDDING_DIM}d]")
    print(f"   📁 Salvos em: {EMBEDDINGS_DIR.resolve()}")

    # 8. Exemplo rápido de similaridade
    print("\n🔍 Top 5 livros para o primeiro usuário:")
    sample_uid  = sorted(user_embeddings.keys())[0]
    u_vec       = user_embeddings[sample_uid]
    book_matrix = np.stack(list(book_embeddings.values()))
    book_keys   = list(book_embeddings.keys())
    scores      = book_matrix @ u_vec
    top5        = np.argsort(scores)[::-1][:5]

    for rank, i in enumerate(top5, 1):
        print(f"   #{rank} book_id={book_keys[i]}  score={scores[i]:.4f}")


if __name__ == "__main__":
    main()