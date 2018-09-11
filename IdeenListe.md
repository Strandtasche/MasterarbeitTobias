
# Ideen

- Features von Schätzungen statt von Messungen
- Timestep als input feature
- onehotencoding für typ und alle schüttgutsorten in ein Netz
- Daten clean up
  - Kollisionstracks aussortieren
  - Fehlerhistogram
- Signifikante Features identifizieren
- Unterschied mit/ohne Orientierung bei Zylinderdatensatz austesten
- Feature Engineering und Tuning
   - Statt x0,...,x4 könnte man die Features so vorverarbeiten, wie das Netz sie wohl auch verarbeiten wird. Z. B.  für die Prädtikion des nächsten Zeitschritts nur die Inkremente "x4 - x3". Ggf. könnten auch höhere Ableitungen integriert werden. Output wäre dann einfach Inkrement zum nächsten Zeitschritt. Bei der Prädiktion zum Düsenbalken käme dann der aktuelle Abstand als zusätzliches Feature rein.
   - Eventuell können dann schlankere Netze verwendet werden, da bspw. keine Neuronen zur Bestimmung der Ableitung notwendig sind. Ggf. könnte man auch überflüssige Neuronen mithilfe von L1-Regularisierung identifizieren.
   - Indem man alles in ähnlicher Größenordnung hat und sich beispielsweise nur eine Abweichung vom aktuellsten Inkrement geben lässt, könnte ggf. eine gute L2-Regularisierung umgesezt werden.
   - Nutzung eines Bias-Neurons
- Unsicherheit der Vorhersage
   - MSE über alle Teilchen gerechnet. Achtung: Abweichung enthält auch Abweichung der Messung (des Labels)
   - Abweichung als Label in neues Netz (Aber: Mehrfachverwendung von Daten)
- Schwierigere Fälle betrachten
   - Aufnahmen von weiter vorne bei der Aufgabe der Partikel
