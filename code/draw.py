import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score
import matplotlib


matplotlib.rcParams['font.family'] = 'Times New Roman'


def plot_cells_and_metacells_tsne(
        Z_sub_raw,        # 原细胞 latent，用于可视化
        y_sub,
        sup_head_cells,   # 细胞分类头
        sup_head_meta,    # 元细胞分类头
        modality_name,
        metacell_ids,
        metacell_centroids,
        k_means_k,
        class_names=None,
        pca_dim=50,
        save_name="separate_tsne.png"
):


    device = next(sup_head_cells.parameters()).device

    # --------------------------------------------------------
    # 1) Label 体系
    # --------------------------------------------------------
    labels_sorted = np.unique(y_sub)
    num_labels = len(labels_sorted)

    if class_names is None:
        class_names = [f"Class {int(l)}" for l in labels_sorted]
    else:
        assert len(class_names) == num_labels

    # label 映射到连续颜色索引
    label2cidx = {l: i for i, l in enumerate(labels_sorted)}
    y_sub_color = np.array([label2cidx[l] for l in y_sub])

    # --------------------------------------------------------
    # 2) 单细胞预测
    # --------------------------------------------------------
    Z_cells_tensor = torch.tensor(Z_sub_raw, dtype=torch.float32, device=device)
    y_pred_cells = sup_head_cells(Z_cells_tensor).argmax(dim=1).cpu().numpy()

    acc_cells = accuracy_score(y_sub, y_pred_cells)
    f1_cells = f1_score(y_sub, y_pred_cells, average='macro')
    y_pred_cells_color = np.array([label2cidx[l] for l in y_pred_cells])

    # --------------------------------------------------------
    # 2.5) 单细胞：指定类别的 per-class accuracy
    # --------------------------------------------------------
    # 与元细胞逻辑完全一致
    # target_cell_classes = [6, 20, 8, 9, 10, 12, 13, 21, 17]  # BMMC
    target_cell_classes = [10, 11, 13, 14]  # isseq

    per_class_acc_cells = {}

    if target_cell_classes is not None:
        target_cell_classes = list(target_cell_classes)

        print("\n[Per-Class Cell Accuracy]")
        for c in target_cell_classes:
            mask = (y_sub == c)

            if mask.sum() == 0:
                acc_c = np.nan
                print(f"  Class {c}: No cells")
            else:
                acc_c = (y_pred_cells[mask] == c).mean()
                cname = class_names[label2cidx[c]] if class_names is not None else str(c)
                print(
                    f"  Class {c} ({cname}): "
                    f"{acc_c * 100:.2f}%  (Cells={mask.sum()})"
                )

            per_class_acc_cells[c] = acc_c


    # --------------------------------------------------------
    # 3) 元细胞标签（多数投票）
    # --------------------------------------------------------
    num_meta = metacell_centroids.shape[0]
    y_meta = np.zeros(num_meta, dtype=y_sub.dtype)
    for i in range(num_meta):
        idx = np.where(metacell_ids == i)[0]
        if idx.size == 0:
            y_meta[i] = np.bincount(y_sub).argmax()
        else:
            y_meta[i] = np.bincount(y_sub[idx]).argmax()

    y_meta_color = np.array([label2cidx[l] for l in y_meta])

    # --------------------------------------------------------
    # 4) 元细胞预测
    # --------------------------------------------------------
    Z_meta_tensor = torch.tensor(metacell_centroids, dtype=torch.float32, device=device)
    y_pred_meta = sup_head_meta(Z_meta_tensor).argmax(dim=1).cpu().numpy()

    acc_meta = accuracy_score(y_meta, y_pred_meta)
    f1_meta = f1_score(y_meta, y_pred_meta, average='macro')

    y_pred_meta_color = np.array([label2cidx[l] for l in y_pred_meta])

    # target_meta_classes = target_cell_classes # BMMC
    target_meta_classes = target_cell_classes # isseq
    per_class_acc_meta = {}
    if target_meta_classes is not None:
        target_meta_classes = list(target_meta_classes)

        print("\n[Per-Class MetaCell Accuracy]")
        for c in target_meta_classes:
            mask = (y_meta == c)

            if mask.sum() == 0:
                acc_c = np.nan
                print(f"  Class {c}: No metacells")
            else:
                acc_c = (y_pred_meta[mask] == c).mean()
                cname = class_names[label2cidx[c]] if class_names is not None else str(c)
                print(f"  Class {c} ({cname}): "
                      f"{acc_c * 100:.2f}%  (MetaCells={mask.sum()})")

            per_class_acc_meta[c] = acc_c

    # --------------------------------------------------------
    # 5) 单细胞 t-SNE
    # --------------------------------------------------------
    pca_cells = PCA(n_components=min(pca_dim, Z_sub_raw.shape[1])).fit_transform(Z_sub_raw)
    tsne_cells = TSNE(
        n_components=2,
        init="pca",
        random_state=0,
        learning_rate="auto",
        perplexity=min(30, max(5, Z_sub_raw.shape[0] // 100))
    ).fit_transform(pca_cells)

    # --------------------------------------------------------
    # 6) 元细胞 t-SNE
    # --------------------------------------------------------
    pca_meta = PCA(n_components=min(pca_dim, metacell_centroids.shape[1])).fit_transform(metacell_centroids)
    tsne_meta = TSNE(
        n_components=2,
        init="pca",
        random_state=0,
        learning_rate="auto",
        perplexity=min(30, max(5, num_meta // 4))
    ).fit_transform(pca_meta)

    # --------------------------------------------------------
    # 7) Color map
    # --------------------------------------------------------
    if num_labels <= 10:
        cmap_base = plt.cm.tab10(np.linspace(0, 1, num_labels))
    else:
        cmap_base = plt.cm.tab20(np.linspace(0, 1, num_labels))
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(cmap_base)

    # --------------------------------------------------------
    # 8) 绘图（整合版，一维 axes）
    # --------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(36, 10))  # 1行4列，直接返回一维 axes
    fig.subplots_adjust(bottom=0.22)  # 底部留空间给 legend
    fig.suptitle(f"{modality_name} | Our |min_samples_per_metacell={k_means_k}", fontsize=30, y=1.05)

    # ---- Color map ----
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(np.unique(y_sub)))]
    num_classes = len(class_names)
    if num_classes <= 10:
        cmap = plt.cm.tab10(np.linspace(0, 1, num_classes))
    elif num_classes <= 20:
        cmap = plt.cm.tab20(np.linspace(0, 1, num_classes))
    else:
        cmap_base = plt.cm.tab20(np.linspace(0, 1, 20))
        extra_n = num_classes - 20
        warm_colors = np.array([plt.cm.autumn(i / max(extra_n - 1, 1))[:3] for i in range(extra_n)])
        warm_colors = np.hstack([warm_colors, np.ones((extra_n, 1))])
        cmap = np.vstack([cmap_base, warm_colors])
    cmap = ListedColormap(cmap)

    axes_fontsize = 25

    # --- 单细胞真值 ---
    axes[0].scatter(tsne_cells[:, 0], tsne_cells[:, 1], c=y_sub_color, s=8, cmap=cmap)
    axes[0].set_title(f"Cells True Labels\nAcc={acc_cells * 100:.2f}%, F1={f1_cells * 100:.2f}%",
                      fontsize=axes_fontsize)

    # --- 单细胞预测 ---
    axes[1].scatter(tsne_cells[:, 0], tsne_cells[:, 1], c=y_pred_cells_color, s=8, cmap=cmap)
    axes[1].set_title(f"Cells Predicted Labels\nAcc={acc_cells * 100:.2f}%, F1={f1_cells * 100:.2f}%",
                      fontsize=axes_fontsize)

    # --- 元细胞真值 ---
    axes[2].scatter(tsne_meta[:, 0], tsne_meta[:, 1], c=y_meta_color, edgecolors="black", s=40, cmap=cmap)
    axes[2].set_title(f"MetaCells True Labels\nAcc={acc_meta * 100:.2f}%, F1={f1_meta * 100:.2f}%",
                      fontsize=axes_fontsize)

    # --- 元细胞预测 ---
    axes[3].scatter(tsne_meta[:, 0], tsne_meta[:, 1], c=y_pred_meta_color, edgecolors="black", s=40, cmap=cmap)
    axes[3].set_title(f"MetaCells Predicted Labels\nAcc={acc_meta * 100:.2f}%, F1={f1_meta * 100:.2f}%",
                      fontsize=axes_fontsize)

    # ---- 不显示刻度 ----
    for ax in axes:
        ax.set_aspect('equal', 'box')
        ax.set_xticks([])
        ax.set_yticks([])

    # --------------------------------------------------------
    # 9) Legend
    # --------------------------------------------------------
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D(
            [0], [0],
            marker='o',
            color='w',
            markerfacecolor=cmap(i),
            markersize=18,
            label=class_names[i]
        )
        for i in range(num_classes)
    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=min(6, num_classes),
        fontsize=25,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02)
    )

    plt.savefig(save_name, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Save] {save_name} generated.")


def plot_metacells_tsne_only(
        Z_sub_raw,        # 仍保留接口一致性（未使用）
        y_sub,
        sup_head_cells,   # 保留接口一致性（未使用）
        sup_head_meta,    # 元细胞分类头
        modality_name,
        metacell_ids,
        metacell_centroids,
        k_means_k,
        class_names=None,
        pca_dim=50,
        save_name="metacell_tsne"
):
    device = next(sup_head_meta.parameters()).device

    # --------------------------------------------------------
    # 1) Label 体系
    # --------------------------------------------------------
    labels_sorted = np.unique(y_sub)
    num_labels = len(labels_sorted)

    if class_names is None:
        class_names = [f"Class {int(l)}" for l in labels_sorted]
    else:
        assert len(class_names) == num_labels

    label2cidx = {l: i for i, l in enumerate(labels_sorted)}

    # --------------------------------------------------------
    # 2) 元细胞标签（多数投票）
    # --------------------------------------------------------
    num_meta = metacell_centroids.shape[0]
    y_meta = np.zeros(num_meta, dtype=y_sub.dtype)

    for i in range(num_meta):
        idx = np.where(metacell_ids == i)[0]
        if idx.size == 0:
            y_meta[i] = np.bincount(y_sub).argmax()
        else:
            y_meta[i] = np.bincount(y_sub[idx]).argmax()

    y_meta_color = np.array([label2cidx[l] for l in y_meta])

    # --------------------------------------------------------
    # 3) 元细胞预测
    # --------------------------------------------------------
    Z_meta_tensor = torch.tensor(
        metacell_centroids,
        dtype=torch.float32,
        device=device
    )
    y_pred_meta = (
        sup_head_meta(Z_meta_tensor)
        .argmax(dim=1)
        .cpu()
        .numpy()
    )

    acc_meta = accuracy_score(y_meta, y_pred_meta)
    f1_meta = f1_score(y_meta, y_pred_meta, average="macro")

    y_pred_meta_color = np.array([label2cidx[l] for l in y_pred_meta])

    # --------------------------------------------------------
    # 4) 元细胞 t-SNE
    # --------------------------------------------------------
    pca_meta = PCA(
        n_components=min(pca_dim, metacell_centroids.shape[1])
    ).fit_transform(metacell_centroids)

    tsne_meta = TSNE(
        n_components=2,
        init="pca",
        random_state=0,
        learning_rate="auto",
        perplexity=min(15, max(5, num_meta // 20)),
        early_exaggeration=24.0
    ).fit_transform(pca_meta)

    # scale = np.sqrt(num_meta*100)  # 元细胞越多，放得越开
    # tsne_meta = tsne_meta * scale

    # --------------------------------------------------------
    # 5) Color map（与原逻辑一致）
    # --------------------------------------------------------
    from matplotlib.colors import ListedColormap

    num_classes = len(class_names)
    if num_classes <= 10:
        cmap_arr = plt.cm.tab10(np.linspace(0, 1, num_classes))
    elif num_classes <= 20:
        cmap_arr = plt.cm.tab20(np.linspace(0, 1, num_classes))
    else:
        base = plt.cm.tab20(np.linspace(0, 1, 20))
        extra_n = num_classes - 20
        warm = np.array([
            plt.cm.autumn(i / max(extra_n - 1, 1))[:3]
            for i in range(extra_n)
        ])
        warm = np.hstack([warm, np.ones((extra_n, 1))])
        cmap_arr = np.vstack([base, warm])

    cmap = ListedColormap(cmap_arr)

    # --------------------------------------------------------
    # 6) 绘图（1行2列）
    # --------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(18, 12))
    fig.subplots_adjust(bottom=0.25)
    fig.suptitle(
        f"{modality_name} Minimum metacell size={k_means_k}",
        fontsize=28,
        fontweight='bold',  # 加粗
        y=1.05
    )

    axes_fontsize = 24

    # --- 元细胞真值 ---
    axes[0].scatter(
        tsne_meta[:, 0],
        tsne_meta[:, 1],
        c=y_meta_color,
        s=45,
        # edgecolors="black",
        cmap=cmap
    )
    axes[0].set_title(
        f"COMET Ground Truth\nAcc={acc_meta*100:.2f}%, F1={f1_meta*100:.2f}%",
        fontsize=axes_fontsize
    )

    # --- 元细胞预测 ---
    axes[1].scatter(
        tsne_meta[:, 0],
        tsne_meta[:, 1],
        c=y_pred_meta_color,
        s=45,
        # edgecolors="black",
        cmap=cmap
    )
    axes[1].set_title(
        f"COMET Predicted Labels\nAcc={acc_meta*100:.2f}%, F1={f1_meta*100:.2f}%",
        fontsize=axes_fontsize
    )

    for ax in axes:
        ax.set_aspect("equal", "box")
        ax.set_xticks([])
        ax.set_yticks([])

    # --------------------------------------------------------
    # 7) Legend（完全一致）
    # --------------------------------------------------------
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor=cmap(i),
            markersize=18,
            label=class_names[i]
        )
        for i in range(num_classes)
    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=min(4, num_classes),
        fontsize=22,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03)
    )

    save_name += str(k_means_k) + '.png'
    plt.savefig(save_name, dpi=1000, bbox_inches="tight")
    plt.close()
    print(f"[Save] {save_name} generated.")

