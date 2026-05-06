# ODM Environment Setup

This project has two orthomosaic-related modes:

- Fast Preview / OpenCV 拼接预览: quick image stitching preview. It is not a professional orthomosaic or GeoTIFF orthorectification result.
- Real ODM Orthomosaic / OpenDroneMap 真实正射处理: local Docker execution of `opendronemap/odm`. Only this mode can produce real ODM outputs such as `odm_orthophoto.tif`.

## Required Local Environment

Real ODM mode requires:

1. Docker Desktop installed and running.
2. The OpenDroneMap image available locally:

```bash
docker pull opendronemap/odm:latest
```

3. UAV aerial images with enough overlap. A practical starting point is 70%-80% overlap.

## Environment Check

In the Gradio tab `正射影像 / 航测拼接预览`, click:

```text
检查 ODM 环境（Docker / 镜像）
```

The check verifies:

- whether Docker is available;
- whether `opendronemap/odm:latest` is available locally.

If either check fails, 灾情感知及影响评估 reports the reason and does not fabricate ODM outputs.

## Expected ODM Outputs

After a successful ODM run, the service checks for:

- `odm_orthophoto/odm_orthophoto.tif`
- `odm_georeferencing/odm_georeferenced_model.laz`
- `odm_georeferencing/odm_georeferenced_model.ply`
- `odm_texturing/odm_textured_model.obj`
- `odm_dem/dsm.tif`
- `odm_dem/dtm.tif`

Some outputs may be missing depending on the input image quality and ODM processing result. The system does not treat every optional output as mandatory.

## Truthfulness Boundary

Do not claim real orthomosaic completion unless `odm_orthophoto.tif` exists.

If ODM runs but does not produce `odm_orthophoto.tif`, the correct status is:

```text
ODM 运行完成，但未找到 odm_orthophoto.tif。
```

If Docker or the ODM image is unavailable, the correct status is:

```text
无法运行真实 ODM 正射处理。
```

Fast Preview output must remain described as an OpenCV stitching preview, not a real orthomosaic.
