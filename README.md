Aquí tienes la traducción del archivo README.md que proporcionaste:

# whisper-timestamped

Reconocimiento automático de habla multilingüe con marcas de tiempo y confiabilidad a nivel de palabra.

* [Descripción](#descripción)
   * [Notas sobre otros enfoques](#notas-sobre-otros-enfoques)
* [Instalación](#instalación)
   * [Primera instalación](#primera-instalación)
      * [Paquetes adicionales que podrían ser necesarios](#paquetes-adicionales-que-podrían-ser-necesarios)
      * [Docker](#docker)
   * [Instalación ligera para CPU](#instalación-ligera-para-cpu)
   * [Actualizar a la última versión](#actualizar-a-la-última-versión)
* [Uso](#uso)
   * [Python](#python)
   * [Línea de comandos](#línea-de-comandos)
   * [Generación de alineación de palabras](#generación-de-alineación-de-palabras)
   * [Ejemplo de salida](#ejemplo-de-salida)
   * [Opciones que pueden mejorar los resultados](#opciones-que-pueden-mejorar-los-resultados)
* [Agradecimientos](#agradecimientos)
* [Citas](#citas)

## Descripción
[Whisper](https://openai.com/blog/whisper/) es un conjunto de modelos de reconocimiento de voz robustos y multilingües entrenados por OpenAI que logran resultados de vanguardia en muchos idiomas. Los modelos de Whisper fueron entrenados para predecir marcas de tiempo aproximadas en segmentos de voz (la mayoría de las veces con una precisión de 1 segundo), pero originalmente no pueden predecir marcas de tiempo de palabras. Este repositorio propone una implementación para **predecir marcas de tiempo de palabras y proporcionar una estimación más precisa de los segmentos de voz al transcribir con modelos de Whisper**.

El enfoque se basa en Dynamic Time Warping (DTW) aplicado a los pesos de atención cruzada, como se demuestra en [este cuaderno de Jong Wook Kim](https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/notebooks/Multilingual_ASR.ipynb). Hay algunas adiciones a este cuaderno:
* La estimación de inicio/fin es más precisa.
* Se asignan puntajes de confianza a cada palabra.
* **Si es posible (sin búsqueda de haz...)**, no se requieren pasos de inferencia adicionales para predecir las marcas de tiempo de las palabras (la alineación de palabras se realiza sobre la marcha después de que se decodifica cada segmento de voz).
* Se ha tenido un cuidado especial en cuanto al uso de la memoria: `whisper-timestamped` es capaz de procesar archivos largos con poca memoria adicional en comparación con el uso regular del modelo Whisper.

`whisper-timestamped` es una extensión del paquete Python [`openai-whisper`](https://pypi.org/project/whisper-openai/) y está destinado a ser compatible con cualquier versión de `openai-whisper`.

### Notas sobre otros enfoques

Un enfoque alternativo relevante para recuperar marcas de tiempo a nivel de palabra implica el uso de modelos wav2vec que predicen caracteres, como se implementa con éxito en [whisperX](https://github.com/m-bain/whisperX). Sin embargo, estos enfoques tienen varias desventajas que no están presentes en los enfoques basados en pesos de atención cruzada como `whisper_timestamped`. Estas desventajas incluyen:
* La necesidad de encontrar un modelo wav2vec por cada idioma a admitir, lo cual no escala bien con las capacidades multilingües de Whisper.
* La necesidad de manejar (al menos) una red neuronal adicional (modelo wav2vec), lo que consume memoria.
* La necesidad de normalizar caracteres en la transcripción de Whisper para que coincidan con el conjunto de caracteres del modelo wav2vec. Esto implica conversiones incómodas dependientes del idioma, como convertir números a palabras ("2" -> "dos"), símbolos a palabras ("%" -> "porcentaje", "€" -> "euro(s)")...
* La falta de robustez en torno a las disfluencias del habla (rellenos, vacilaciones, palabras repetidas...) que suelen eliminarse en Whisper.

Un enfoque alternativo que no requiere un modelo adicional es observar las probabilidades de los tokens de marca de tiempo estimadas por el modelo Whisper después de que se predice cada (sub)palabra. Esto se implementó, por ejemplo, en whisper.cpp y stable-ts. Sin embargo, este enfoque carece de robustez porque los modelos Whisper no han sido entrenados para generar marcas de tiempo significativas después de cada palabra. Los modelos Whisper tienden a predecir marcas de tiempo solo después de que se hayan predicho un cierto número de palabras (normalmente al final de una oración), y la distribución de probabilidad de marcas de tiempo fuera de esta condición puede ser inexacta. En la práctica, estos métodos pueden producir resultados que están completamente fuera de sincronización en algunos períodos de tiempo (lo observamos especialmente cuando hay música de jingle). Además, la precisión de las marcas de tiempo de los modelos Whisper tiende a redondearse a 1 segundo (como en muchos subtítulos de video), lo que es demasiado inexacto para las palabras y lograr una mayor precisión es complicado.

## Instalación

### Primera instalación

Requisitos:
* `python3` (versión igual o superior a 3.7, se recomienda al menos 3.9)
* `ffmpeg` (consultar las instrucciones de instalación en el [repositorio de Whisper](https://github.com/openai/whisper))

Puedes instalar `whisper-timestamped` utilizando pip:
```bash
pip3 install git+https://github.com/linto-ai/whisper-timestamped
```

o clonando este repositorio y ejecutando la instalación:
```bash
git clone https://github.com/linto-ai/whisper-timestamped
cd whisper-timestamped/
python3 setup.py install
```

#### Paquetes adicionales que podrían ser necesarios

Si deseas graficar la alineación entre las marcas de tiempo de audio y las palabras (como se muestra en [esta sección](#plotting-word-alignment)), también necesitas matplotlib:
```bash
pip3 install matplotlib
```

Si deseas usar la opción VAD (

Detección de Actividad de Voz antes de ejecutar el modelo Whisper), también necesitas torchaudio y onnxruntime:
```bash
pip3 install onnxruntime torchaudio
```

Si deseas usar modelos Whisper ajustados finamente desde Hugging Face Hub, también necesitas transformers:
```bash
pip3 install transformers
```

#### Docker

Se puede construir una imagen de Docker de aproximadamente 9GB utilizando:
```bash
git clone https://github.com/linto-ai/whisper-timestamped
cd whisper-timestamped/
docker build -t whisper_timestamped:latest .
```

### Instalación ligera para CPU

Si no tienes una GPU (o no deseas usarla), entonces no necesitas instalar las dependencias de CUDA. Debes instalar una versión ligera de torch **antes** de instalar whisper-timestamped, por ejemplo, de la siguiente manera:
```bash
pip3 install \
     torch==1.13.1+cpu \
     torchaudio==0.13.1+cpu \
     -f https://download.pytorch.org/whl/torch_stable.html
```

También se puede construir una imagen de Docker específica de aproximadamente 3.5GB de la siguiente manera:
```bash
git clone https://github.com/linto-ai/whisper-timestamped
cd whisper-timestamped/
docker build -t whisper_timestamped_cpu:latest -f Dockerfile.cpu .
```

### Actualizar a la última versión

Cuando uses pip, la biblioteca se puede actualizar a la última versión utilizando:
```
pip3 install --upgrade --no-deps --force-reinstall git+https://github.com/linto-ai/whisper-timestamped
```

Una versión específica de `openai-whisper` se puede utilizar ejecutando, por ejemplo:
```bash
pip3 install openai-whisper==20230124
```

## Uso

### Python

En Python, puedes usar la función `whisper_timestamped.transcribe()`, que es similar a la función `whisper.transcribe()`:
```python
import whisper_timestamped
help(whisper_timestamped.transcribe)
```
La principal diferencia con `whisper.transcribe()` es que la salida incluirá una clave `"words"` para todos los segmentos, con el inicio y final de la palabra. Ten en cuenta que la palabra incluirá signos de puntuación. Consulta el ejemplo [a continuación](#ejemplo-de-salida).

Además, las opciones de decodificación predeterminadas son diferentes para favorecer la decodificación eficiente (decodificación codiciosa en lugar de búsqueda de haz y sin muestreo de temperatura de respaldo). Para tener las mismas opciones predeterminadas que en `whisper`, utiliza ```beam_size=5, best_of=5, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)```.

También hay opciones adicionales relacionadas con la alineación de palabras.

En general, si importas `whisper_timestamped` en lugar de `whisper` en tu script de Python y usas `transcribe(model, ...)` en lugar de `model.transcribe(...)`, debería funcionar:
```
import whisper_timestamped as whisper

audio = whisper.load_audio("AUDIO.wav")

model = whisper.load_model("tiny", device="cpu")

result = whisper.transcribe(model, audio, language="fr")

import json
print(json.dumps(result, indent=2, ensure_ascii=False))
```

Ten en cuenta que puedes utilizar un modelo Whisper ajustado finamente desde HuggingFace o una carpeta local utilizando el método `load_model` de `whisper_timestamped`. Por ejemplo, si deseas usar [whisper-large-v2-nob](https://huggingface.co/NbAiLab/whisper-large-v2-nob), simplemente puedes hacer lo siguiente:
```
import whisper_timestamped as whisper

model = whisper.load_model("NbAiLab/whisper-large-v2-nob", device="cpu")

# ...
```

### Línea de comandos

También puedes usar `whisper_timestamped` en la línea de comandos, de manera similar a `whisper`. Consulta la ayuda con:
```bash
whisper_timestamped --help
```

Las principales diferencias con la CLI de `whisper` son:
* Archivos de salida:
  * El JSON de salida contiene marcas de tiempo de palabras y puntajes de confianza. Consulta el ejemplo [a continuación](#ejemplo-de-salida).
  * Hay un formato de salida adicional en CSV.
  * Para los formatos SRT, VTT, TSV, se guardarán archivos adicionales con marcas de tiempo de palabras.
* Algunas opciones predeterminadas son diferentes:
  * Por defecto, no se establece una carpeta de salida: Usa `--output_dir .` para el valor predeterminado de Whisper.
  * Por defecto, no hay modo verbose: Usa `--verbose True` para el valor predeterminado de Whisper.
  * Por defecto, la decodificación de búsqueda de haz y el muestreo de temperatura de respaldo están deshabilitados para favorecer una decodificación eficiente.
    Para configurar lo mismo que el valor predeterminado de Whisper, puedes usar `--accurate` (que es un alias de ```--beam_size 5 --temperature_increment_on_fallback 0.2 --best_of 5```).
* Hay algunas opciones adicionales específicas:
  <!-- * `--efficient` para usar una decodificación codiciosa más rápida (sin búsqueda de haz ni varios muestreos en cada paso),
  lo que habilita un camino especial donde las marcas de tiempo de palabras se calculan sobre la marcha (no es necesario ejecutar la inferencia dos veces).
  Ten en cuenta que los resultados de la transcripción pueden ser significativamente peores en grabaciones desafiantes con esta opción. -->
  * `--compute_confidence` para habilitar/deshabilitar el cálculo de puntajes de confianza para cada palabra.
  * `--punctuations_with_words` para decidir si los signos de puntuación deben incluirse o no con las palabras precedentes.

Un ejemplo de comando para procesar varios archivos usando el modelo `tiny` y guardar los resultados en la carpeta actual, como se haría por defecto con Whisper, es el siguiente:
```
whisper_timestamped audio1.flac audio2.mp3 audio3.wav --model tiny --output_dir .
```

Ten en cuenta que puedes usar un modelo Whisper ajustado finamente desde HuggingFace o una carpeta local. Por ejemplo, si deseas usar el modelo [whisper-large-v2-nob](https://huggingface.co/NbAiLab/whisper-large-v2-nob), simplemente puedes hacer lo siguiente:
```
whisper_timestamped --model NbAiLab/whisper-large-v2-nob <...>
```

### Generación de alineación de palabras



Ten en cuenta que puedes usar la opción `plot_word_alignment` de la función `whisper_timestamped.transcribe()` en Python o la opción `--plot` de la CLI `whisper_timestamped` para ver la alineación de palabras para cada segmento.

![Ejemplo de alineación](figs/example_alignement_plot.png)

* El gráfico superior representa la transformación de pesos de atención cruzada utilizados para la alineación con Dynamic Time Warping. El eje de las abscisas representa el tiempo y el de las ordenadas representa los tokens predichos, con tokens de marca de tiempo especiales al principio y al final, y (sub)palabras y puntuación en el medio.
* El gráfico inferior es una representación de MFCC de la señal de entrada (características utilizadas por Whisper, basadas en el cepstrum de frecuencia de Mel).
* Las líneas verticales punteadas en rojo muestran dónde se encuentran los límites de las palabras (con signos de puntuación "pegados" a la palabra anterior).

### Ejemplo de salida

Aquí tienes un ejemplo de salida de la función `whisper_timestamped.transcribe()`, que se puede ver utilizando la CLI:
```bash
whisper_timestamped AUDIO_FILE.wav --model tiny --language fr
```
```json
{
  "text": " Bonjour! Est-ce que vous allez bien?",
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.5,
      "end": 1.2,
      "text": " Bonjour!",
      "tokens": [ 25431, 2298 ],
      "temperature": 0.0,
      "avg_logprob": -0.6674491882324218,
      "compression_ratio": 0.8181818181818182,
      "no_speech_prob": 0.10241222381591797,
      "confidence": 0.51,
      "words": [
        {
          "text": "Bonjour!",
          "start": 0.5,
          "end": 1.2,
          "confidence": 0.51
        }
      ]
    },
    {
      "id": 1,
      "seek": 200,
      "start": 2.02,
      "end": 4.48,
      "text": " Est-ce que vous allez bien?",
      "tokens": [ 50364, 4410, 12, 384, 631, 2630, 18146, 3610, 2506, 50464 ],
      "temperature": 0.0,
      "avg_logprob": -0.43492694334550336,
      "compression_ratio": 0.7714285714285715,
      "no_speech_prob": 0.06502953916788101,
      "confidence": 0.595,
      "words": [
        {
          "text": "Est-ce",
          "start": 2.02,
          "end": 3.78,
          "confidence": 0.441
        },
        {
          "text": "que",
          "start": 3.78,
          "end": 3.84,
          "confidence": 0.948
        },
        {
          "text": "vous",
          "start": 3.84,
          "end": 4.0,
          "confidence": 0.935
        },
        {
          "text": "allez",
          "start": 4.0,
          "end": 4.14,
          "confidence": 0.347
        },
        {
          "text": "bien?",
          "start": 4.14,
          "end": 4.48,
          "confidence": 0.998
        }
      ]
    }
  ],
  "language": "fr"
}
```

### Opciones que pueden mejorar los resultados

Aquí hay algunas opciones que no están habilitadas de forma predeterminada pero que pueden mejorar los resultados.

#### Transcripción precisa de Whisper

Como se mencionó anteriormente, algunas opciones de decodificación están deshabilitadas de forma predeterminada para ofrecer una mejor eficiencia. Sin embargo, esto puede afectar la calidad de la transcripción. Para ejecutar con las opciones que tienen la mejor probabilidad de proporcionar una buena transcripción, utiliza las siguientes opciones.
* En Python:
```python
results = whisper_timestamped.transcribe(model, audio, beam_size=5, best_of=5, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), ...)
```
* En la línea de comandos:
```bash
whisper_timestamped --accurate ...
```

#### Ejecutar la Detección de Actividad de Voz (VAD) antes de enviar a Whisper

Los modelos Whisper pueden "alucinar" texto cuando se les proporciona un segmento sin voz. Esto se puede evitar ejecutando VAD y uniendo segmentos de voz antes de transcribir con el modelo Whisper. Esto es posible con `whisper-timestamped`.
* En Python:
```python
results = whisper_timestamped.transcribe(model, audio, vad=True, ...)
```
* En la línea de comandos:
```bash
whisper_timestamped --vad True ...
```

#### Detección de disfluencias

Los modelos Whisper tienden a eliminar las disfluencias del habla (palabras de relleno, vacilaciones, repeticiones, etc.). Sin precauciones, las disfluencias que no se transcriben afectarán la marca de tiempo de la palabra siguiente: la marca de tiempo del inicio de la palabra será en realidad la marca de tiempo del inicio de las disfluencias. `whisper-timestamped` puede tener algunas heurísticas para evitar esto.
* En Python:
```python
results = whisper_timestamped.transcribe(model, audio, detect_disfluencies=True, ...)
```
* En la línea de comandos:
```bash
whisper_timestamped --detect_disfluencies True ...
```
**Importante:** Ten en cuenta que al usar estas opciones, es posible que las disfluencias potenciales aparezcan en la transcripción como una palabra especial "`[*]`".


## Acknowlegment
* [whisper](https://github.com/openai/whisper): Whisper speech recognition (License MIT).
* [dtw-python](https://pypi.org/project/dtw-python): Dynamic Time Warping (License GPL v3).

## Citations
If you use this in your research, please cite the repo:

```bibtex
@misc{lintoai2023whispertimestamped,
  title={whisper-timestamped},
  author={Louradour, J{\'e}r{\^o}me},
  journal={GitHub repository},
  year={2023},
  publisher={GitHub},
  howpublished = {\url{https://github.com/linto-ai/whisper-timestamped}}
}
```

as well as the OpenAI Whisper paper:

```bibtex
@article{radford2022robust,
  title={Robust speech recognition via large-scale weak supervision},
  author={Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  journal={arXiv preprint arXiv:2212.04356},
  year={2022}
}
```

and this paper for Dynamic-Time-Warping:

```bibtex
@article{JSSv031i07,
  title={Computing and Visualizing Dynamic Time Warping Alignments in R: The dtw Package},
  author={Giorgino, Toni},
  journal={Journal of Statistical Software},
  year={2009},
  volume={31},
  number={7},
  doi={10.18637/jss.v031.i07}
}
```
