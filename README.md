# Proyecto de Mejora de Imágenes Confocales con CycleGAN + SwinIR

Este repositorio contiene la implementación de un esquema no pareado tipo CycleGAN donde:
- El generador encargado de la transformación de **baja resolución (LR)** a **alta resolución (HR)** está basado en **SwinIR**.
- El generador para la transformación inversa (HR a LR) es una arquitectura ResNet más ligera.
- Se incluyen discriminadores para cada dominio (LR y HR) y las pérdidas correspondientes (adversarial, ciclo, identidad).

El objetivo principal es **mejorar la calidad** de imágenes confocales, aprovechando un dataset **no pareado** de imágenes LR y HR (ambas de 256×256 píxeles).

## Estructura del proyecto

La estructura de carpetas y archivos es la siguiente:

```bash
TuProyecto/
├── README.md
├── config.py
├── data/
│   └── dataset.py
├── models/
│   ├── generator_swinir.py
│   ├── resnet_generator.py
│   ├── discriminator.py
│   └── cyclegan_model.py
├── utils/
│   ├── losses.py
│   └── metrics.py
├── train.py
└── test.py
