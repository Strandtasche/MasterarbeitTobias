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

### Training/Evaluation

1. sicherstellen, dass das richtige virtual environment aktiviert ist.
2. in den `implementation` Ordner wechseln.
3.  `python DNNRegressor-Example.py [args]` mit den richtigen Argumenten. 


### Beispiele

Für das Trainieren eines Netzes:
```bash
python DNNRegressor-Example.py --training --custom --augment --filter --save ~/data/pfefferRutscheSeparator-FS5-filterAugm-750Distance.h5 --hyperparams ~/filteredParticles-Real-Pfeffer-Rutsche-final.json --overrideModel ~/models/Separator/pfefferRutscheSeparator 2>&1 | tee -a ~/logs/RealPfeffer_Rutsche_Final.txt
```


Für die Evaluation eines Netzes mit Plots:
```bash
python DNNRegressor-Example.py --plot --lossAna --custom --load ~/data/pfefferRutscheSeparator-FS5-filterAugm-750Distance.h5 --hyperparams ~/filteredParticles-Real-Pfeffer-Rutsche-final.json --overrideModel ~/models/Separator/pfefferRutscheSeparator 2>&1 | tee -a ~/logs/RealPfeffer_Rutsche_Final.txt
```



## Kommandozeilenparameter


`--training`: Wenn diese Flag übergeben wird, werden die Gewichte des Modells upgedatet. Wenn nicht wird evaluiert.<br/>
`--single`: Die Trennung zwischen Trainings- und Testdatenset wird aufgehoben. Verwenden für Validation. <br/>
`--plot`: Das Ergebnis der Netze wird mit matplotlib geplottet.<br/>
`--lossAna`: Plottet die schlechtesten Prädiktionen des Testsets.<br/>
`--plotNo PLOTNO`: Die Anzahl der Feature-Label-Paare, die exemplarisch betrachtet werden sollen. Default=10, type=int.<br/>
`--hyperparams PARAMS`: Die Hyperparameter-Datei, die für die Ausführung benutzt werden soll. Default="hyper_params.json", type=str.<br/>
`--dispWeights`: Wenn diese Flag gesetzt wird, werden die Gewichte der einzelnen Neuronenverbindungen ausgegeben und eine grafische Darstellung der Gewichte zwischen dem Input Layer und dem ersten Hidden Layer angezeigt.<br/>
`--target TARGET`: Flag, die einen Ziel-MSE angibt, bei dessem erreichen/unterschreiten das Training abgebrochen wird. type=float.<br/>
`--custom`: Flag, die dafür sorgt, dass der Custom Estimator statt dem Premade Estimator benutzt wird. Sollte eigentlich immer verwendet werden, da die meisten Verbesserungen für diesen geschrieben wurden. 


`--save '*'`: Flag, die bestimmt ob und wo, die bestimmte Feature-Label-Paare von Trainings- und Testset gespeichert werden sollen. Wenn kein Dateipfad angegeben wird, wird als Default in `data.h5` im ModelPath gespeichert. Mutually exclusive mit `--load`.<br/>
`--load '*'`: Flag, die bestimmt ob und von wo, die bestimmte Feature-Label-Paare von Trainings- und Testset geladen werden sollen. Wenn kein Dateipfad angegeben wird, wird als Default versucht aus `data.h5` im ModelPath zu laden. Mutually exclusive mit `--load`. Wenn diese Flag gesetzt ist werden keine neuen Feature-Label-Paare aus den CSV-Dateien generier.<br/>


`--overrideModel MODELFOLDER`: Flag, die den im Hyperparameter File vorgegebenen Modell-Ordner überschreibt. <br/>
`--overrideInput FILEPATH`: Flag, die den im Hyperparameter File vorgegebenen Dateipfad überschreibt. <br/>
`--separator POSITION`: Flag, die den Separator Parameter und die Position aus dem Hyperparameter File überschreibt. type=int. Use with care.

`--augment`: Flag, die festlegt, ob beim generieren des Trainingssets data augmentation verwendet werden soll.<br/>
`--filter POSITION`: Flag, die die Feature-Label-Paare so filtert, dass von der gegebenen Position wepredictet wird.

`--tensorboard_debug_address ADDRESS`: Connect to the TensorBoard Debugger Plugin backend specified by the gRPC address (e.g., localhost:1234). Mutually exclusive with the `--debug` flag.



**Flags that are not usable right now:**
`--progressplot`: Flag, die dafür sorgt, dass die Prädiktionen für ein einzelnes Feature-Label-Paar zu verschiedenen Zeitpunkten während dem Training dargestellt wird.
`--fake FAKE`: Flag, die dafür sorgt, das ausschließlich synthetische Daten verwendet werden.
`--debug`: Flag, die ursprüngliche Debug Funktionen aktiviert



## License

TODO?





