# MasterarbeitTobias
Masterarbeit Sommersemester 2018, am KIT - ISAS, Karlsruhe
Thema: Ableitung von Bewegungsmodellen für Anwendungen in der Schüttgutsortierung mittels Machine Learning

betreut von Florian Pfaff und Georg Maier.

## Abstact 
> Die Erweiterung von optischen Schüttgutsortierern mit Flächenkameras ermöglicht eine Verbesserung der Sortierqualität durch bessere Bewegungsprädiktion.
> Dafür wird ein Modell benötigt, das die Bewegung der einzelnen Schüttgutpartikel gut abbildet.
> Ein solches Bewegungsmodell zu bestimmen, ist aufwendig und verschiedene Bewegungsmodelle erreichen bei unterschiedlichen Schüttgütern variierende > Qualität.
> 
> Im Rahmen dieser Arbeit wurden neuronale Netze eingesetzt, um Bewegungsmodelle für verschiedene Schüttgüter zu ermitteln.
> Dazu wurden mehrere Datensätze für ausgewählte Schüttgüter 
> am modularen Schüttgutsortiersystem _TableSort_ des Fraunhofer IOSBs in zwei Konfigurationen erfasst und für das Training vorverarbeitet.
> Es wurde ein für die Problemstellung passendes Datenaugmentierungsverfahren implementiert und geeignete Hyperparameter für das Training der > verschiedenen Szenarien ermittelt.
> 
> Die Evaluation wurde sowohl auf den so ermittelten als auch auf bestehenden, mittels der _Diskrete-Elemente-Methode_ simulierten Datensätzen > durchgeführt.
> In allen Szenarien waren die Ergebnisse der neuronalen Netze vergleichbar oder besser als die der betrachteten Referenzverfahren.
> Insbesondere auf den Realdaten konnte eine Verbesserung der Sortierqualität im Vergleich zu den Referenzverfahren erzielt werden. 

## Getting Started

in diesem Abschnitt wird beschrieben, wie man den Projekt benutzt.

### Voraussetzungen

Die Wichtigste von allen Voraussetzungen ist [TensorFlow](https://www.tensorflow.org/).
Die Evaluation der Ergebnisse ist unter TensorFlow 1.11 durchgeführt worden.
Wahrscheinlich funktioniert es auch mit Versionen > 1.8 und insbesondere mit neueren Versionen. 

Im Rahmen dieser Arbeit wurde TensorFlow mittels pip installiert in eine virtual environment installiert.
Dies ist immer noch eine Option, inzwischen wird von den Entwicklern eine Installation über einen [Docker Container](https://www.tensorflow.org/install/docker) empfohlen, 
da so der zugegebenermaßen aufwändige Prozess um GPU Unterstützung zu erhalten vereinfacht wird.
Für eine Evaluation der existierenden Netze ist die GPU Unterstützung wahrscheinlich nicht notwendig, 
aber um eigene Netze zu trainieren wäre es empfehlenswert.

Eine komplette Liste der Python Packages, die in der virtual environment installiert wurden ist in der [requirements.txt]{implementation/requirements-gpu-upgraded.txt} Datei zu finden.

### Training

1. sicherstellen, dass das richtige virtual environment aktiviert ist.
2. in den `implementation` Ordner wechseln
3.  `richtigen Befehl`

### Evaluation


1. sicherstellen, dass das richtige virtual environment aktiviert ist.
2. in den `implementation` Ordner wechseln
3.  `richtigen Befehl`


## Kommandozeilenparameter

TODO: überlegen wie ich das formatiere

## License



##




