import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F

# 读 h5ad
import scanpy as sc
from scipy.sparse import issparse
import pandas as pd
import os

os.environ["OMP_NUM_THREADS"] = "1"



# =========================================================
# 0. 设备选择：优先 MPS -> CUDA -> CPU
# =========================================================

# if torch.backends.mps.is_available():
    # DEFAULT_DEVICE = "mps"
if torch.cuda.is_available():
    DEFAULT_DEVICE = "cuda"
else:
    DEFAULT_DEVICE = "cpu"

print(f"[Device] 使用设备: {DEFAULT_DEVICE}")


# =========================================================
# 1. 直接加载你的 Zheng68k 数据
# =========================================================
def load_user_data_new() -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """
    直接从 h5ad 文件读取 RNA 数据和标签：
        X_all: (N_cells, N_genes) float32
        y_all: (N_cells,) int64, 0~C-1
        class_names: {int -> str} 映射
    标签列默认在 obs['cell_type1']
    """
    h5ad_path = '../data/Zheng68k_PBMC_1024.h5ad'
    label_col = "cell_type1"  # 从 h5ad.obs 读取

    print(f"[Load] 读取 h5ad: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)

    # --- X 数据 ---
    if issparse(adata.X):
        X_all = adata.X.toarray()
    else:
        X_all = adata.X

    # --- label ---
    if label_col not in adata.obs.columns:
        raise KeyError(f"[Load] h5ad obs 中找不到列 {label_col}，现有列：{adata.obs.columns.tolist()}")

    labels = adata.obs[label_col].astype("category")
    y_all = labels.cat.codes.to_numpy(dtype=np.int64)

    # 分类名称映射
    class_names_list = list(labels.cat.categories)
    class_names = {i: name for i, name in enumerate(class_names_list)}

    # 转为 np.array
    X_all = np.asarray(X_all, dtype=np.float32)
    y_all = np.asarray(y_all, dtype=np.int64)

    print(f"[Load] X_all.shape={X_all.shape}, y_all.shape={y_all.shape}")
    print(f"[Load] 发现 {len(class_names)} 个细胞类型 (从 0 编码到 {len(class_names)-1})")
    print("[Load] 细胞类型列表:")
    for i, name in class_names.items():
        print(f"   id={i:2d} -> {name}")

    return X_all, y_all, class_names


def load_user_data() -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """
    读取你的 Zheng68k_PBMC_1024.h5ad，并返回：
        X_all: (N_cells, N_genes) float32
        y_all: (N_cells,) int64, 0~C-1
        class_names: {int -> str} 映射
    """
    h5ad_path = "./bmmc/rna.h5ad"
    label_col = "x"
    label_csv_path = "./bmmc/Label.csv"

    print(f"[Load] 读取 h5ad: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)

    if issparse(adata.X):
        X_all = adata.X.toarray()
    else:
        X_all = adata.X

    # --- 读取 label CSV ---
    n_cells = X_all.shape[0]

    df_label = pd.read_csv(label_csv_path)
    print(f"[Load] label csv shape={df_label.shape}, columns={df_label.columns.tolist()}")

    # 确保 label_col 存在
    if label_col not in df_label.columns:
        raise KeyError(f"[Load] label CSV 中找不到列 {label_col}，现有列：{df_label.columns.tolist()}")

    # 检查行数一致（无 cell_id 必须完全一致）
    if len(df_label) != n_cells:
        raise ValueError(
            f"[Error] label CSV 行数 {len(df_label)} 与 h5ad 细胞数 {n_cells} 不一致，无法按顺序匹配。"
        )

    # 直接按顺序读取 label
    labels = df_label[label_col].astype("category")
    y_all = labels.cat.codes.to_numpy(dtype=np.int64)

    # 分类名称映射
    class_names_list = list(labels.cat.categories)
    class_names = {i: name for i, name in enumerate(class_names_list)}

    X_all = np.asarray(X_all, dtype=np.float32)
    y_all = np.asarray(y_all, dtype=np.int64)

    print(f"[Load] X_all.shape={X_all.shape}, y_all.shape={y_all.shape}")
    print(f"[Load] 发现 {len(class_names)} 个细胞类型 (从 0 编码到 {len(class_names) - 1})")
    print("[Load] 细胞类型列表:")
    for i, name in class_names.items():
        print(f"   id={i:2d} -> {name}")

    return X_all, y_all, class_names


def build_metacell_latent(metacell_ids, Z_all, y_all, model):
    """
    返回：
        meta_labels: (K,)
        meta_latent: (K, latent_dim)
        meta_shallow: (K, shallow_dim)
        meta_deep: (K, deep_dim)
    """

    shallow_dim = model.shallow_dim
    latent_dim = model.latent_dim

    # --------------------------------------------------------
    # 1) 元细胞总数 K = 码本总维度（不是 unique）
    # --------------------------------------------------------
    K = sum(cb.num_embeddings for cb in model.codebooks)
    print("Total metacells K =", K)

    # --------------------------------------------------------
    # 2) 投票生成元细胞标签 (K,)
    # --------------------------------------------------------
    meta_labels = np.zeros(K, dtype=int)

    from collections import Counter
    for m in range(K):
        mask = (metacell_ids == m)
        if mask.sum() == 0:
            # 未使用元细胞 → -1
            meta_labels[m] = -1
        else:
            counts = Counter(y_all[mask])
            meta_labels[m] = counts.most_common(1)[0][0]

    # --------------------------------------------------------
    # 3) 计算浅变量均值 (K, shallow_dim)
    # --------------------------------------------------------
    meta_shallow = np.zeros((K, shallow_dim), dtype=np.float32)

    for m in range(K):
        mask = (metacell_ids == m)
        if mask.sum() == 0:
            continue
        meta_shallow[m] = Z_all[mask, :shallow_dim].mean(axis=0)

    # --------------------------------------------------------
    # 4) 提取 codebook weight 作为 deep variable (K, deep_dim)
    # --------------------------------------------------------
    meta_deep_list = []
    for cb in model.codebooks:
        meta_deep_list.append(cb.weight.detach().cpu().numpy())
    meta_deep = np.concatenate(meta_deep_list, axis=0)  # (K, deep_dim)

    # --------------------------------------------------------
    # 5) 拼接 shallow + deep = meta latent
    # --------------------------------------------------------
    meta_latent = np.concatenate([meta_shallow, meta_deep], axis=1)

    # --- 移除 label = -1 的元细胞 ---
    valid_mask = (meta_labels != -1)
    meta_labels = meta_labels[valid_mask]
    meta_latent = meta_latent[valid_mask]
    meta_deep = meta_deep[valid_mask]

    return meta_labels, meta_latent, meta_shallow, meta_deep


# =========================================================
# 2. Class-Conditional VQ-AE （在 AE 里面做元细胞量化）
# =========================================================

class ClassConditionalVQVAE(nn.Module):
    """
    实际为 Shared-Codebook VQ-VAE
    （接口保持 class-conditional 兼容，用于消融）
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        # codebook_sizes: List[int],  # 仍接收，但只用 sum
        total_codes: int,
        latent_dim: int = 32,

        hidden_dims: Tuple[int, ...] = (512, 256),
        beta: float = 0.25,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.beta = beta

        # self.total_codes = sum(codebook_sizes)
        self.total_codes = total_codes
        # ===== 兼容旧 pipeline：共享 codebook 下 offset 恒为 0 =====
        self.register_buffer(
            "codebook_offsets",
            torch.zeros(self.n_classes, dtype=torch.long)
        )

        # Encoder
        enc_layers = []
        last_dim = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(last_dim, h), nn.ReLU()]
            last_dim = h
        enc_layers.append(nn.Linear(last_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = []
        last_dim = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(last_dim, h), nn.ReLU()]
            last_dim = h
        dec_layers.append(nn.Linear(last_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        # ===== 共享 codebook =====
        self.codebook = nn.Embedding(self.total_codes, latent_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / self.total_codes, 1.0 / self.total_codes)

        # ===== 兼容旧 pipeline：伪造一个 codebooks =====
        # 所有“类 codebook”其实都指向同一个 embedding
        self.codebooks = nn.ModuleList(
            [self.codebook for _ in range(n_classes)]
        )

    def _quantize_shared(
        self,
        z_e: torch.Tensor   # (B, D)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Shared codebook quantization
        """
        device = z_e.device
        B, D = z_e.shape

        E = self.codebook.weight            # (K, D)

        z_norm2 = (z_e ** 2).sum(dim=1, keepdim=True)     # (B, 1)
        e_norm2 = (E ** 2).sum(dim=1).unsqueeze(0)        # (1, K)
        dist = z_norm2 + e_norm2 - 2.0 * z_e @ E.t()      # (B, K)

        code_idx = dist.argmin(dim=1)       # (B,)
        z_q = E[code_idx]                   # (B, D)

        # usage loss（鼓励 code 均匀）
        counts = torch.bincount(code_idx, minlength=self.total_codes).float()
        probs = counts / (counts.sum() + 1e-8)
        valid = probs > 0
        if valid.any():
            p = probs[valid]
            usage_loss = (p * torch.log(p * p.numel())).sum()
        else:
            usage_loss = torch.tensor(0.0, device=device)

        return z_q, code_idx, usage_loss

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor    # y 保留但不使用
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        z_e = self.encoder(x)

        z_q, code_idx_local, usage_loss = self._quantize_shared(z_e)

        vq_loss = (
            F.mse_loss(z_q.detach(), z_e) +
            self.beta * F.mse_loss(z_q, z_e.detach())
        )

        z_st = z_e + (z_q - z_e).detach()
        x_rec = self.decoder(z_st)



        return x_rec, vq_loss, usage_loss, code_idx_local

    @torch.no_grad()
    def encode_quantized(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        z_e = self.encoder(x)
        z_q, code_idx_local, _ = self._quantize_shared(z_e)
        z_st = z_e + (z_q - z_e).detach()

        return z_st, code_idx_local



@dataclass
class VQVAETrainingConfig:
    latent_dim: int = 32
    hidden_dims: Tuple[int, ...] = (512, 256)
    batch_size: int = 512
    lr: float = 1e-3
    max_epochs: int = 200
    weight_decay: float = 1e-5
    val_split: float = 0.1
    early_stop_patience: int = 25
    beta: float = 0.25
    usage_reg_weight: float = 1e-3   # usage 正则的系数
    device: str = DEFAULT_DEVICE

    # codebook 大小控制：每个 code 约 ≈ target_cells_per_code 个细胞
    target_cells_per_code: int = 15
    min_codes_per_class: int = 8
    max_codes_per_class: int = 256


def auto_codebook_sizes_per_class(
    y_all: np.ndarray,
    n_classes: int,
    cfg: VQVAETrainingConfig
) -> List[int]:
    """根据每个类的细胞数量自动估计各类 codebook 大小。"""
    counts = np.bincount(y_all, minlength=n_classes)
    sizes = []
    print("[VQVAE] 每个类的细胞数:")
    for c in range(n_classes):
        print(f"  class {c}: N={counts[c]}")
        k = int(round(counts[c] / cfg.target_cells_per_code))
        k = max(cfg.min_codes_per_class, min(cfg.max_codes_per_class, k))
        sizes.append(k)

    print("[VQVAE] 自动设置每类 codebook 大小:")
    for c, k in enumerate(sizes):
        print(f"  class {c}: codebook_size={k}")
    print(f"[VQVAE] 总 code数量 = {sum(sizes)}")
    return sizes


def train_vqvae_on_cells(
    X_all: np.ndarray,
    y_all: np.ndarray,
    n_classes: int,
    gene_scaler: StandardScaler,
    cfg: Optional[VQVAETrainingConfig] = None
) -> ClassConditionalVQVAE:
    if cfg is None:
        cfg = VQVAETrainingConfig()

    device = cfg.device
    print(f"[VQVAE] 训练设备: {device}")

    # 标准化（已在外面 fit，这里 transform）
    X_scaled = gene_scaler.transform(X_all).astype(np.float32)

    X_t = torch.from_numpy(X_scaled)
    y_t = torch.from_numpy(y_all.astype(np.int64))

    dataset = TensorDataset(X_t, y_t)
    n_total = len(dataset)
    n_val = max(1, int(cfg.val_split * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    print(f"[VQVAE] 数据: total={n_total}, train={n_train}, val={n_val}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    input_dim = X_all.shape[1]
    codebook_sizes = auto_codebook_sizes_per_class(y_all, n_classes, cfg)

    model = ClassConditionalVQVAE(
        input_dim=input_dim,
        n_classes=n_classes,
        # codebook_sizes=codebook_sizes,
        total_codes=sum(codebook_sizes),
        latent_dim=cfg.latent_dim,
        hidden_dims=cfg.hidden_dims,
        beta=cfg.beta,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    criterion_recon = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        train_recon_sum = 0.0
        train_vq_sum = 0.0
        train_usage_sum = 0.0
        n_train_samples = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            x_rec, vq_loss, usage_loss, _ = model(batch_x, batch_y)
            recon_loss = criterion_recon(x_rec, batch_x)
            total_loss = recon_loss + vq_loss + cfg.usage_reg_weight * usage_loss

            total_loss.backward()
            optimizer.step()

            bs = batch_x.size(0)
            n_train_samples += bs
            train_recon_sum += recon_loss.item() * bs
            train_vq_sum += vq_loss.item() * bs
            train_usage_sum += usage_loss.item() * bs

        train_recon = train_recon_sum / n_train_samples
        train_vq = train_vq_sum / n_train_samples
        train_usage = train_usage_sum / n_train_samples

        # ---------- 验证 ----------
        model.eval()
        val_recon_sum = 0.0
        val_vq_sum = 0.0
        val_usage_sum = 0.0
        n_val_samples = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                x_rec, vq_loss, usage_loss, _ = model(batch_x, batch_y)
                recon_loss = criterion_recon(x_rec, batch_x)

                bs = batch_x.size(0)
                n_val_samples += bs
                val_recon_sum += recon_loss.item() * bs
                val_vq_sum += vq_loss.item() * bs
                val_usage_sum += usage_loss.item() * bs

        val_recon = val_recon_sum / n_val_samples
        val_vq = val_vq_sum / n_val_samples
        val_usage = val_usage_sum / n_val_samples
        val_total = val_recon + val_vq + cfg.usage_reg_weight * val_usage

        print(
            f"[VQVAE] Epoch {epoch:03d}/{cfg.max_epochs} | "
            f"train_recon={train_recon:.6f} | train_vq={train_vq:.6f} | train_usage={train_usage:.6f} || "
            f"val_recon={val_recon:.6f} | val_vq={val_vq:.6f} | val_usage={val_usage:.6f} | "
            f"val_total={val_total:.6f}"
        )

        if val_total < best_val_loss - 1e-5:
            best_val_loss = val_total
            best_epoch = epoch
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stop_patience:
                print(
                    f"[VQVAE] 早停触发: 连续 {cfg.early_stop_patience} 轮无提升，"
                    f"使用第 {best_epoch} 轮的参数 (val_total={best_val_loss:.6f})"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    print("[VQVAE] 训练完成。")
    return model


# =========================================================
# 3. 用训练好的 VQ-AE 编码所有细胞 -> latent + 元细胞 ID
# =========================================================

def encode_cells_vq(
    X_all: np.ndarray,
    y_all: np.ndarray,
    model: ClassConditionalVQVAE,
    gene_scaler: StandardScaler,
    batch_size: int = 1024,
    device: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回:
        Z_all: (N_cells, latent_dim) 量化后的 latent
        metacell_ids: (N_cells,) 全局元细胞 ID（按类+code idx 展开）
    """
    if device is None:
        device = DEFAULT_DEVICE

    model.eval()
    model.to(device)

    X_scaled = gene_scaler.transform(X_all).astype(np.float32)
    X_t = torch.from_numpy(X_scaled)
    y_t = torch.from_numpy(y_all.astype(np.int64))

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    all_z = []
    all_mc = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            z_q, code_idx_local = model.encode_quantized(batch_x, batch_y)

            # 全局 metacell id = offset[class] + local_idx
            offsets = model.codebook_offsets[batch_y].to(device)  # (B,)
            mc_global = offsets + code_idx_local

            all_z.append(z_q.cpu().numpy())
            all_mc.append(mc_global.cpu().numpy())

    Z_all = np.concatenate(all_z, axis=0)
    metacell_ids = np.concatenate(all_mc, axis=0)

    print(f"[EncodeVQ] 所有细胞已编码到 latent 空间，Z_all.shape={Z_all.shape}")
    print(f"[EncodeVQ] 元细胞 ID 范围: {metacell_ids.min()} ~ {metacell_ids.max()}, "
          f"总数约={int(metacell_ids.max() + 1)}")
    return Z_all, metacell_ids


# =========================================================
# 4. t-SNE 对比图：原空间 vs VQ latent
# =========================================================

# 假设你已经有：
# Z_all: 单细胞 latent
# metacell_ids: 新元细胞 ID
# metacell_centroids: 每个元细胞中心
def plot_tsne_compare(
    X: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    class_names: dict,
    metacell_ids: np.ndarray = None,
    metacell_centroids: np.ndarray = None,
    max_points: int = 10000,
    random_state: int = 0,
    pca_dim: int = 50,
    save_prefix: str = None
):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import numpy as np

    rng = np.random.default_rng(random_state)
    n_cells = X.shape[0]

    # -------------------
    # 1) 随机抽样
    # -------------------
    if n_cells > max_points:
        idx = rng.choice(n_cells, size=max_points, replace=False)
        X_sub, Z_sub, y_sub = X[idx], Z[idx], y[idx]
        if metacell_ids is not None:
            metacell_ids_sub = metacell_ids[idx]
        else:
            metacell_ids_sub = None
    else:
        X_sub, Z_sub, y_sub = X, Z, y
        metacell_ids_sub = metacell_ids

    classes = np.unique(y_sub)
    cmap = plt.get_cmap("tab20")
    colors = {c: cmap(i % 20) for i, c in enumerate(classes)}

    # -------------------
    # 2) PCA
    # -------------------
    if pca_dim is not None and X_sub.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=random_state)
        X_for_tsne = pca.fit_transform(X_sub)
    else:
        X_for_tsne = X_sub

    # -------------------
    # 3) t-SNE
    # -------------------
    tsne_raw = TSNE(n_components=2, random_state=random_state, init="pca", learning_rate="auto")
    X_tsne = tsne_raw.fit_transform(X_for_tsne)

    tsne_latent = TSNE(n_components=2, random_state=random_state, init="pca", learning_rate="auto")
    Z_tsne = tsne_latent.fit_transform(Z_sub)

    # -------------------
    # 4) 绘图
    # -------------------
    plt.figure(figsize=(18, 6))

    # 左图：原始
    plt.subplot(1, 3, 1)
    for c in classes:
        mask = (y_sub == c)
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                    s=5, alpha=0.6, label=class_names[c], c=[colors[c]])
    plt.title("t-SNE on raw (or PCA(raw))")
    plt.legend(markerscale=3, fontsize=8)

    # 中图：VQ latent
    plt.subplot(1, 3, 2)
    for c in classes:
        mask = (y_sub == c)
        plt.scatter(Z_tsne[mask, 0], Z_tsne[mask, 1],
                    s=5, alpha=0.6, label=class_names[c], c=[colors[c]])
    plt.title("t-SNE on VQ-AE latent")
    plt.legend(markerscale=3, fontsize=8)

    # 右图：元细胞
    # 右图：元细胞
    # 右图：元细胞（稳健版：通过 nearest-centroid 在 latent 空间推断 y_meta）
    if metacell_ids_sub is not None and metacell_centroids is not None:
        # Z_sub: 单细胞 latent, shape (Ns, D)
        # y_sub: 单细胞 label, shape (Ns,)
        # metacell_centroids: (Nm, D)
        # metacell_ids_sub: (Ns,) 每个单细胞对应的元细胞 id（可能不是 0..Nm-1）

        # 1) 计算每个单细胞到每个 centroid 的最近邻（按欧氏距离）
        from sklearn.neighbors import NearestNeighbors

        Nm = metacell_centroids.shape[0]
        Ns = Z_sub.shape[0]

        # 用 NearestNeighbors 找出每个单细胞最接近的元细胞索引（0..Nm-1）
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(metacell_centroids)
        distances, nn_idx = nbrs.kneighbors(Z_sub, return_distance=True)  # nn_idx: (Ns, 1)
        nn_idx = nn_idx.ravel()  # (Ns,)

        # 2) 为每个 centroid 做多数投票确定它的 class
        y_meta = np.zeros(Nm, dtype=y_sub.dtype)
        for i in range(Nm):
            assign_mask = (nn_idx == i)
            if assign_mask.sum() == 0:
                # 没有任何单细胞最近邻到该 centroid，fallback：用全局最接近的单细胞类别
                # find nearest single cell by distance to centroid
                dists_to_cells = np.sum((Z_sub - metacell_centroids[i:i + 1]) ** 2, axis=1)
                nearest_cell_idx = dists_to_cells.argmin()
                y_meta[i] = y_sub[nearest_cell_idx]
            else:
                y_meta[i] = np.bincount(y_sub[assign_mask]).argmax()

        # 3) 拼接 single cells + meta centroids，用 y_meta（可靠）作为元细胞标签
        Z_combined = np.vstack([Z_sub, metacell_centroids])
        y_combined = np.concatenate([y_sub, y_meta])
        is_meta = np.concatenate([np.zeros(len(Z_sub), dtype=bool),
                                  np.ones(len(metacell_centroids), dtype=bool)])

        # 4) 固定颜色映射：把颜色绑定到 class id（而不是出现顺序）
        classes_all = np.unique(y_combined)
        cmap = plt.get_cmap("tab20")
        colors_all = {int(c): cmap(int(c) % 20) for c in classes_all}

        # 5) t-SNE 并绘图（单细胞小点，元细胞大点带黑边）
        print("[t-SNE] 开始 t-SNE on single cells + MetaCells (robust mapping)...")
        plt.subplot(1, 3, 3)
        tsne_combined = TSNE(n_components=2, random_state=random_state,
                             init="pca", learning_rate="auto")
        Z_combined_tsne = tsne_combined.fit_transform(Z_combined)

        for c in sorted(classes_all):
            # 单细胞
            mask_cell = (y_combined == c) & (~is_meta)
            if mask_cell.any():
                plt.scatter(
                    Z_combined_tsne[mask_cell, 0],
                    Z_combined_tsne[mask_cell, 1],
                    s=5,
                    alpha=0.6,
                    c=[colors_all[int(c)]],
                    label=class_names.get(int(c), f"class {int(c)}")
                )

            # 元细胞
            mask_meta = (y_combined == c) & (is_meta)
            if mask_meta.any():
                plt.scatter(
                    Z_combined_tsne[mask_meta, 0],
                    Z_combined_tsne[mask_meta, 1],
                    s=15,
                    c=[colors_all[int(c)]],
                    edgecolors="black",
                    linewidths=1.2,
                    alpha=0.95
                )

        plt.title("t-SNE: single cells + MetaCells (robust)")
        plt.legend(markerscale=3, fontsize=8)

    plt.tight_layout()
    if save_prefix is not None:
        fname = f"{save_prefix}_tsne_compare_metacell.png"
        plt.savefig(fname, dpi=300)
        print(f"[t-SNE] 图已保存到 {fname}")
    else:
        plt.show()


# =========================================================
# 5. Few-shot 原型分类 + 监督分类头
# =========================================================

def build_prototypes(Z_support: np.ndarray, y_support: np.ndarray) -> Dict[Any, np.ndarray]:
    prototypes = {}
    for y in np.unique(y_support):
        idx = np.where(y_support == y)[0]
        z_mean = Z_support[idx].mean(axis=0)
        prototypes[y] = z_mean
    return prototypes


def predict_with_prototypes(Z_query: np.ndarray, prototypes: Dict[Any, np.ndarray]) -> np.ndarray:
    labels = list(prototypes.keys())
    proto_mat = np.stack([prototypes[y] for y in labels], axis=0)

    Z_norm2 = (Z_query ** 2).sum(axis=1, keepdims=True)
    P_norm2 = (proto_mat ** 2).sum(axis=1, keepdims=True).T
    cross = Z_query @ proto_mat.T
    dist2 = Z_norm2 + P_norm2 - 2 * cross

    idx_min = dist2.argmin(axis=1)
    y_pred = np.array([labels[i] for i in idx_min])
    return y_pred


class ClassifierHead(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, z):
        return self.net(z)


@dataclass
class SupHeadTrainingConfig:
    batch_size: int = 32
    lr: float = 1e-3
    max_epochs: int = 500
    weight_decay: float = 1e-4
    early_stop_patience: int = 80
    device: str = DEFAULT_DEVICE


def train_supervised_head_11way10shot(
    Z_support: np.ndarray,
    y_support: np.ndarray,
    Z_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    cfg: Optional[SupHeadTrainingConfig] = None
) -> ClassifierHead:
    if cfg is None:
        cfg = SupHeadTrainingConfig()

    device = cfg.device
    latent_dim = Z_support.shape[1]

    X_t = torch.from_numpy(Z_support.astype(np.float32))
    y_t = torch.from_numpy(y_support.astype(np.int64))
    train_loader = DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=cfg.batch_size,
        shuffle=True
    )

    X_val_t = torch.from_numpy(Z_val.astype(np.float32)).to(device)
    y_val_np = y_val.copy()

    model = ClassifierHead(latent_dim=latent_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    print(f"[SupHead] few-shot 训练: support={len(Z_support)}, val(query)={len(Z_val)}, "
          f"classes={num_classes}")
    print(f"[SupHead] 配置: batch_size={cfg.batch_size}, max_epochs={cfg.max_epochs}, "
          f"early_stop_patience={cfg.early_stop_patience}, device={device}")

    best_state = None
    best_val_acc = -1.0
    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        loss_sum = 0.0
        n_total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()

            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            bs = batch_x.size(0)
            loss_sum += loss.item() * bs
            n_total += bs

        train_loss = loss_sum / n_total

        # 验证（query 集）
        model.eval()
        with torch.no_grad():
            logits_val = model(X_val_t)
            y_pred_val = logits_val.argmax(dim=1).cpu().numpy()
        val_acc = (y_pred_val == y_val_np).mean()

        print(f"[SupHead] Epoch {epoch:03d}/{cfg.max_epochs} "
              f"| train_loss={train_loss:.6f} | val_acc={val_acc * 100:.2f}%")

        if val_acc > best_val_acc + 1e-5:
            best_val_acc = val_acc
            best_state = model.state_dict()
            patience_counter = 0
            best_epoch = epoch
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stop_patience:
                print(
                    f"[SupHead] 早停触发: 连续 {cfg.early_stop_patience} 轮无提升，"
                    f"使用第 {best_epoch} 轮的参数 (val_acc={best_val_acc * 100:.2f}%)"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def eval_supervised_head(
    model: ClassifierHead,
    Z: np.ndarray,
    y: np.ndarray,
    device: Optional[str] = None
) -> float:
    if device is None:
        device = DEFAULT_DEVICE
    model.eval()
    model.to(device)
    X_t = torch.from_numpy(Z.astype(np.float32)).to(device)
    with torch.no_grad():
        logits = model(X_t)
        y_pred = logits.argmax(dim=1).cpu().numpy()
    acc = (y_pred == y).mean()
    return acc


import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple


def refine_metacells(
    metacell_id: np.ndarray,   # (N_cells,) 原始 codebook/global ID
    z_all: np.ndarray,         # (N_cells, latent_dim) latent 特征
    min_samples_per_metacell: int = 6,
    max_metacells_per_cluster: int = 50
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    基于原始 metacell_id 和 latent 特征生成动态元细胞：
        - 样本数少的簇保持一个元细胞
        - 样本数多的簇用 KMeans 分成多个元细胞
        - 保证每个元细胞样本数 >= min_samples_per_metacell

    返回：
        new_metacell_ids: (N_cells,) 每个样本对应的新元细胞 ID（全局唯一）
        metacell_centroids: (N_metacells, latent_dim) 每个元细胞中心
    """
    N, D = z_all.shape
    new_metacell_ids = np.zeros(N, dtype=np.int64)
    centroids_list = []
    cid2new: Dict[int, list] = {}
    global_counter = 0
    unique_clusters = np.unique(metacell_id)

    # ---- 单次 refine，生成新的 metacell ----
    for cid in unique_clusters:

        mask = (metacell_id == cid)
        z_c = z_all[mask]  # 当前 cluster 的所有 latent
        n_samples = z_c.shape[0]

        # 记录此 cluster 在新 metacell ID 空间中的起点
        start_id = global_counter

        # 动态决定子元细胞数量
        n_meta = max(1, min(n_samples // min_samples_per_metacell, max_metacells_per_cluster))

        if n_meta == 1:
            # 少量样本：均值作为中心
            centroid = z_c.mean(axis=0)
            centroids_list.append(centroid)
            new_metacell_ids[mask] = global_counter
            global_counter += 1
        else:
            # 多样本：KMeans 拆分
            kmeans = KMeans(n_clusters=n_meta, random_state=0, n_init='auto')
            labels = kmeans.fit_predict(z_c)

            # 原始索引
            idx = np.where(mask)[0]

            for i in range(n_meta):
                sub_mask = (labels == i)
                sub_idx = idx[sub_mask]

                centroids_list.append(kmeans.cluster_centers_[i])

                new_metacell_ids[sub_idx] = global_counter
                global_counter += 1

        # 记录此 cluster 对应的新 metacell ID 范围
        cid2new[cid] = list(range(start_id, global_counter))

    # ---- 最终输出 ----
    metacell_centroids = np.vstack(centroids_list)

    print(f"[RefineMetacells] 原始 {len(unique_clusters)} 簇 → "
          f"生成 {metacell_centroids.shape[0]} 个元细胞")

    return new_metacell_ids, metacell_centroids, cid2new


def hvg_variance_streaming(
    X,
    n_top_genes=2000,
    eps=1e-8,
):
    """
    X: np.ndarray (cells, genes), float32 / float64
    返回:
        gene_idx
    """

    n_cells, n_genes = X.shape

    gene_mean = np.zeros(n_genes, dtype=np.float64)
    gene_var = np.zeros(n_genes, dtype=np.float64)

    # 按 gene 维度循环（外层 genes，内层 cells）
    for g in range(n_genes):
        xg = X[:, g]              # 这是 view，不拷贝
        xg = np.log1p(xg)         # 只产生一个 (cells,) 的临时向量
        gene_mean[g] = xg.mean()
        gene_var[g] = xg.var()

        if g % 1000 == 0:
            print(f"HVG progress: {g}/{n_genes}")

    dispersion = gene_var / (gene_mean + eps)
    gene_idx = np.argsort(dispersion)[-n_top_genes:]

    return gene_idx


# =========================================================
# 6. 完整流程：VQ-AE 元细胞 + 11-way 10-shot few-shot
# =========================================================

def run_full_pipeline_11way10shot():
    # 1) 加载数据
    X_all, y_all, class_names = load_user_data()

    from sklearn.model_selection import train_test_split

    X_train, X_rest, y_train, y_rest = train_test_split(
        X_all,
        y_all,
        train_size=0.5,
        random_state=42,
        shuffle=True,
        stratify=y_all
    )

    print(f"Train: {X_train.shape}, Rest: {X_rest.shape}")

    # 可选：检查类别分布
    import numpy as np
    print("Train class counts:", np.bincount(y_train))
    print("Rest  class counts:", np.bincount(y_rest))


    is_Zei = False
    if is_Zei:
        # Zeisei需要先HVG筛选
        gene_idx = hvg_variance_streaming(X_all, n_top_genes=1024)
        X_all = np.log1p(X_all[:, gene_idx])

    n_cells, n_genes = X_all.shape
    classes = np.unique(y_all)
    n_classes = len(classes)
    print(f"[Data] X_all.shape={X_all.shape}, y_all.shape={y_all.shape}, "
          f"classes={n_classes}, cells={n_cells}")

    # 2) 基因方向标准化
    gene_scaler = StandardScaler(with_mean=True, with_std=True)
    gene_scaler.fit(X_all)
    print("[Scaler] gene_scaler 已拟合。")

    # 3) 在全体细胞上训练 Class-Conditional VQ-AE
    save_path = "./vq_shared_pretrained.pth"
    device = DEFAULT_DEVICE

    if os.path.exists(save_path):
        print(f"[Load] 检测到预训练权重 {save_path}，直接加载")

        checkpoint = torch.load(save_path, map_location=device)

        # 恢复模型结构
        model_cfg_loaded = checkpoint["config"]
        vq_model = ClassConditionalVQVAE(
            input_dim=model_cfg_loaded["input_dim"],
            n_classes=model_cfg_loaded["n_classes"],
            total_codes=model_cfg_loaded["total_codes"],
            latent_dim=model_cfg_loaded["latent_dim"],
            hidden_dims=tuple(model_cfg_loaded["hidden_dims"]),
            beta=model_cfg_loaded["beta"],
        ).to(device)

        # 加载权重
        vq_model.load_state_dict(checkpoint["model_state"])
        vq_model.eval()

        # 加载 scaler
        gene_scaler = checkpoint["scaler_state"]

    else:
        print("[Train] 未检测到预训练权重，开始训练 VQ-VAE...")

        # ===== 使用你提供的训练配置 =====
        vq_cfg = VQVAETrainingConfig(
            latent_dim=64,
            hidden_dims=(512, 256),
            batch_size=512,
            lr=1e-3,
            max_epochs=200,
            weight_decay=1e-5,
            val_split=0.1,
            early_stop_patience=25,
            beta=0.25,
            usage_reg_weight=1,
            device=device,
            target_cells_per_code=10,
            min_codes_per_class=24,
            max_codes_per_class=256,
        )

        # 训练模型
        vq_model = train_vqvae_on_cells(
            X_train, y_train, n_classes, gene_scaler, vq_cfg
        )

        # ===== 保存权重 =====
        torch.save({
            "model_state": vq_model.state_dict(),
            "scaler_state": gene_scaler,
            "config": {
                "input_dim": X_all.shape[1],
                "n_classes": n_classes,
                "total_codes": vq_model.total_codes,
                "latent_dim": vq_cfg.latent_dim,
                "hidden_dims": vq_cfg.hidden_dims,
                "beta": vq_cfg.beta,
                "usage_reg_weight": vq_cfg.usage_reg_weight,
            },
        }, save_path)

        print(f"[Save] 训练完成，已保存到: {save_path}")

    # 4) 所有细胞编码到 VQ latent 空间
    Z_all, metacell_ids = encode_cells_vq(X_all, y_all, vq_model, gene_scaler,
                                          batch_size=1024, device=DEFAULT_DEVICE)

    print(np.unique(metacell_ids))
    # print(Z_all.shape[0])
    # 假设你已有 metacell_id 和 z_all
    min_cell = 10
    new_metacell_ids, metacell_centroids, cid2new = refine_metacells(
        metacell_id=metacell_ids,
        z_all=Z_all,
        min_samples_per_metacell=min_cell,
        max_metacells_per_cluster=50
    )
    unique_count = np.unique(Z_all, axis=0).shape[0]
    print("Unique vectors:", unique_count, " / ", Z_all.shape[0])

    print("新元细胞 ID 范围:", new_metacell_ids.min(), "~", new_metacell_ids.max())
    print("元细胞 embedding shape:", metacell_centroids.shape)

    '''
    # 5) t-SNE 对比：原空间 vs VQ latent
    plot_tsne_compare(
        X=X_all,
        Z=Z_all,
        y=y_all,
        class_names=class_names,
        metacell_ids=new_metacell_ids,
        metacell_centroids=metacell_centroids,
        save_prefix="fig1"
    )
    '''

    num_metacells = metacell_centroids.shape[0]
    # 6) 构造 11-way 10-shot （每类选 10 个支持，其余为 query）
    y_meta = np.zeros(num_metacells, dtype=y_all.dtype)

    for i in range(num_metacells):
        mask = (new_metacell_ids == i)
        if mask.sum() == 0:
            # 空元细胞（理论上少见） → fallback：给一个全局最常见类
            # 也可以用 -1 或者随机选一个
            y_meta[i] = np.bincount(y_all).argmax()
            continue

        # 多数投票
        y_meta[i] = np.bincount(y_all[mask]).argmax()

    # --- 2) Few-shot: 使用元细胞 ---
    rng = np.random.default_rng(0)
    k_shot = 3

    # 正确：元细胞编号 0 ~ N-1
    meta_indices = np.arange(num_metacells)

    classes = np.unique(y_meta)
    print("[Few-shot] 可用类别:", classes)

    support_idx = []
    query_idx = []

    num = 0
    for c in classes:
        # 正确：选取属于类别 c 的所有 “元细胞编号”
        idx_c = meta_indices[y_meta == c]

        rng.shuffle(idx_c)

        if len(idx_c) <= k_shot:
            print(f"[Warning] 类别 {c} 只有 {len(idx_c)} 个元细胞，跳过")
            continue

        support_idx.extend(idx_c[:k_shot])
        query_idx.extend(idx_c[k_shot:])
        num += 1

    print(f'num={num}')
    support_idx = np.array(support_idx)
    query_idx = np.array(query_idx)

    Z_support = metacell_centroids[support_idx]
    y_support = y_meta[support_idx]
    Z_query = metacell_centroids[query_idx]
    y_query = y_meta[query_idx]

    print(f"[Few-shot] 支持集 = {Z_support.shape[0]} 个元细胞")
    print(f"[Few-shot] 查询集 = {Z_query.shape[0]} 个元细胞")

    # 7) 原型 few-shot baseline
    prototypes = build_prototypes(Z_support, y_support)
    y_pred_proto = predict_with_prototypes(Z_query, prototypes)
    acc_proto = (y_pred_proto == y_query).mean()
    print(f"[Proto] 原型分类准确率: {acc_proto * 100:.2f}%")

    # 8) 监督分类头 few-shot 训练
    sup_cfg = SupHeadTrainingConfig(
        batch_size=32,
        lr=1e-3,
        max_epochs=500,
        weight_decay=1e-4,
        early_stop_patience=80,
        device=DEFAULT_DEVICE,
    )
    sup_head = train_supervised_head_11way10shot(
        Z_support,
        y_support,
        Z_query,
        y_query,
        num_classes=n_classes,
        cfg=sup_cfg
    )

    acc_sup = eval_supervised_head(sup_head, Z_query, y_query, device=DEFAULT_DEVICE)
    print(f"[SupHead] 监督分类头最终在查询集的准确率: {acc_sup * 100:.2f}%")

    # -----------------------------
    # Few-shot on single-cells (对照实验)
    # -----------------------------
    print("\n================= 单细胞 Few-shot（对照） =================")

    # 单细胞用 y_all（原始标签）
    labels_cell = y_all
    cell_indices = np.arange(len(y_all))  # 0 ~ N_cells-1
    rng = np.random.default_rng(0)

    classes_cell = np.unique(labels_cell)
    print("[Cell Few-shot] 可用类别:", classes_cell)

    support_idx_cells = []
    query_idx_cells = []
    num_cell_classes = 0

    for c in classes_cell:
        idx_c = cell_indices[labels_cell == c]

        rng.shuffle(idx_c)

        if len(idx_c) <= k_shot:
            print(f"[Cell Warning] 类别 {c} 只有 {len(idx_c)} 个原始细胞，跳过")
            continue

        support_idx_cells.extend(idx_c[:k_shot])
        query_idx_cells.extend(idx_c[k_shot:])
        num_cell_classes += 1

    support_idx_cells = np.array(support_idx_cells)
    query_idx_cells = np.array(query_idx_cells)

    print(f"[Cell Few-shot] classes used={num_cell_classes}, "
          f"support={len(support_idx_cells)}, query={len(query_idx_cells)}")

    # ----------- 用 "原始输入 X_all" 做单细胞 few-shot（对照） -----------
    X_support_cells = X_all[support_idx_cells]
    y_support_cells = y_all[support_idx_cells]
    X_query_cells = X_all[query_idx_cells]
    y_query_cells = y_all[query_idx_cells]

    # 原型 baseline（注意：这里必须也是用 X_all，否则不一致）
    prototypes_cells = build_prototypes(X_support_cells, y_support_cells)
    y_pred_proto_cells = predict_with_prototypes(X_query_cells, prototypes_cells)
    acc_proto_cells = (y_pred_proto_cells == y_query_cells).mean()
    print(f"[Cell Proto] 原型分类准确率: {acc_proto_cells * 100:.2f}%")

    # 用 "原始 X_all" 训练监督分类头
    sup_head_cells = train_supervised_head_11way10shot(
        X_support_cells, y_support_cells,
        X_query_cells, y_query_cells,
        num_classes=n_classes, cfg=sup_cfg
    )

    acc_sup_cells = eval_supervised_head(
        sup_head_cells, X_query_cells, y_query_cells, device=device
    )
    print(f"[Cell SupHead] 单细胞 few-shot 查询集准确率: {acc_sup_cells * 100:.2f}%")

    from draw import plot_cells_and_metacells_tsne
    plot_cells_and_metacells_tsne(
        Z_sub_raw=X_all,
        y_sub=y_all,
        sup_head_cells=sup_head_cells,   # 单细胞 head
        sup_head_meta=sup_head,     # 元细胞 head
        modality_name='Zheng68k',
        metacell_ids=new_metacell_ids,
        metacell_centroids=metacell_centroids,
        k_means_k=min_cell,
        class_names=class_names,
        save_name="fig_cells_meta_tsne.png")

    return {
        "X_all": X_all,
        "y_all": y_all,
        "class_names": class_names,
        "gene_scaler": gene_scaler,
        "vq_model": vq_model,
        "Z_all": Z_all,
        "metacell_ids": metacell_ids,
        "support_idx": support_idx,
        "query_idx": query_idx,
        "Z_support": Z_support,
        "y_support": y_support,
        "Z_query": Z_query,
        "y_query": y_query,
        "prototypes": prototypes,
        "acc_proto": acc_proto,
        "sup_head": sup_head,
        "acc_sup": acc_sup,
    }


if __name__ == "__main__":
    results = run_full_pipeline_11way10shot()