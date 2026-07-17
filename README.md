# Rescue-Net — SLAM collaboratif et sémantique pour la recherche et le sauvetage

Déployer une flotte de micro-drones et de robots terrestres dans une zone de catastrophe (séisme, incendie) pour reconstruire une **carte sémantique ultra-légère** exploitable instantanément par les secours.

Le SLAM classique produit des nuages de points massifs, impossibles à transmettre via les réseaux saturés d'une zone sinistrée. Avec l'**Object SLAM**, les robots détectent et géolocalisent uniquement des entités clés (victime, bonbonne de gaz, pilier effondré, issue bloquée) et échangent un dictionnaire d'objets 3D de quelques kilo-octets.

**Phase 1 (ce dépôt)** : pipeline offline mono-robot — ORB-SLAM3 + YOLO → carte objets 3D JSON + visualisation Open3D.

## Architecture

```
Dataset (TUM / EuRoC / vidéo)
    → ORB-SLAM3 (poses caméra)
    → YOLOv8-nano (détection sémantique)
    → Projection 2D→3D (RGB-D ou mono)
    → Fusion multi-vues
    → object_map.json + viewer Open3D
```

Voir [docs/architecture.md](docs/architecture.md) pour le détail.

## Installation

### Python

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### ORB-SLAM3 (optionnel pour Phase 1)

Voir [slam/README.md](slam/README.md). Sans ORB-SLAM3, utilisez `--identity-poses` ou fournissez un `poses.txt` existant.

## Datasets

Configurer les chemins dans `config/datasets.yaml` :

| Dataset | Téléchargement | Usage |
|---------|----------------|-------|
| **TUM RGB-D** `freiburg1_xyz` | https://vision.in.tum.de/data/datasets/rgbd-dataset/download | Priorité — depth réelle |
| **EuRoC** `MH_01_easy` | https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets | Mono + trajectoire drone |
| **Vidéo sinistre** | Placer un MP4 dans `data/videos/` | Démo qualitative |

Exemple TUM :

```bash
mkdir -p data/tum
# Extraire rgbd_dataset_freiburg1_xyz sous data/tum/
```

## Utilisation

### Pipeline complet (TUM RGB-D)

```bash
# 1. SLAM (si ORB-SLAM3 est compilé)
./slam/run_slam.sh tum data/tum/rgbd_dataset_freiburg1_xyz output/poses.txt

# 2. Sémantique + carte objets
python scripts/run_pipeline.py \
  --dataset tum \
  --dataset-root data/tum/rgbd_dataset_freiburg1_xyz \
  --poses output/poses.txt \
  --stride 10 \
  --no-view
```

Tout-en-un avec SLAM intégré :

```bash
python scripts/run_pipeline.py \
  --dataset tum \
  --dataset-root data/tum/rgbd_dataset_freiburg1_xyz \
  --run-slam \
  --stride 10
```

### Sans SLAM (test rapide)

```bash
python scripts/run_pipeline.py \
  --dataset tum \
  --dataset-root data/tum/rgbd_dataset_freiburg1_xyz \
  --identity-poses \
  --max-frames 50 \
  --stride 5 \
  --no-view
```

### EuRoC / vidéo

```bash
python scripts/run_pipeline.py --dataset euroc --dataset-root data/euroc/MH_01_easy/mav0 --poses output/poses.txt
python scripts/run_pipeline.py --dataset video --dataset-root data/videos/disaster_demo.mp4 --identity-poses
```

## Sorties

| Fichier | Description |
|---------|-------------|
| `output/object_map.json` | Dictionnaire d'objets 3D (~quelques Ko) |
| `output/map_snapshot.ply` | Snapshot trajectoire + objets (Open3D) |
| `output/poses.txt` | Trajectoire caméra format TUM |

Exemple d'entrée dans `object_map.json` :

```json
{
  "class_name": "victim",
  "position": [1.2, 0.5, 3.1],
  "confidence": 0.87,
  "observations": 4
}
```

## Classes sémantiques MVP

| Classe Rescue-Net | Proxy COCO |
|-------------------|------------|
| `victim` | person |
| `blocked_exit` | door |
| `collapsed` | chair, couch, bed |
| `fire_source` | fire hydrant |
| `gas_cylinder` | *(modèle custom — Phase 2)* |

Config : `config/semantic/classes.yaml`

## Structure du projet

```
Rescue-Net/
├── config/          # datasets, SLAM, classes sémantiques
├── slam/            # ORB-SLAM3 wrapper
├── semantic/        # détection YOLO + projection 3D
├── mapping/         # ObjectMap + fusion
├── viz/             # viewer Open3D
├── scripts/         # run_pipeline.py
└── docs/            # architecture
```

## Roadmap

- **Phase 1** (actuelle) : mono-robot offline, carte objets JSON
- **Phase 2** : export protobuf + sync gRPC multi-agent
- **Phase 3** : mesh P2P / ROS2
- **Phase 4** : visualisation AR poste secours

## Stack technique cible

ORB-SLAM sémantique · protocoles Mesh/P2P · Edge AI ultra-léger · gRPC (Phase 2+)

## Licence

À définir. ORB-SLAM3 : GPLv3. YOLOv8 : AGPL-3.0 (Ultralytics).
