= Training
Die Datensätze, die für das Training verwendet wurden, sind das eganscha/gomoku_vlm_ds Dataset, die Repository für den Trainingscode und Dataset generation befindet sich unter github.com/frederik-uni/robo-muehle, und als Basis-Modell diente Qwen/Qwen3-VL-2B-Instruct.

== Überarbeitung des Trainingsskripts
Das ursprüngliche Trainingsskript wies mehrere Fehler und Inkonsistenzen auf, die zu instabilen Trainingsläufen und unzuverlässigen Evaluierungen führten. Insbesondere waren Teile des Datensatz preprocessing fehlerhaft implementiert. Aus diesem Grund wurde das Skript gänzlich überarbeitet.

== Trainingskonfiguration und Hyperparameterwahl

Das Fine-Tuning des Modells erfolgte mittels Supervised Fine-Tuning (SFT) unter Verwendung von Low-Rank Adaptation (LoRA), um eine parameter-effiziente Anpassung großer Sprach- und Multimodalmodelle zu ermöglichen. Ziel war es, sowohl visuelle als auch logische Fähigkeiten schrittweise zu verbessern, ohne das Basismodell vollständig neu zu trainieren.

== Optimierungsstrategie

Als Optimierer kam AdamW (adamw_torch) mit einer Gewichtsnormierung (weight decay) von 0.01 zum Einsatz. Die Lernrate wurde mittels eines Cosine Learning Rate Schedulers mit einer Warmup-Phase von 10 % der gesamten Trainingsschritte gesteuert. Diese Kombination erwies sich als stabil und reduzierte Oszillationen im Trainingsverlauf.

Die maximale Gradienten-Norm wurde auf 1.0 begrenzt, um exploding Gradient zu vermeiden. Das Training wurde im bfloat16-Modus durchgeführt, was eine höhere numerische Stabilität gegenüber fp16 bei vergleichbarem Speicherverbrauch bietet.

== Epochen, Batch-Größe und Akkumulation

Das Training wurde für 3 Epochen durchgeführt. Die effektive Batch-Größe ergab sich aus der Kombination von per_device_train_batch_size und gradient_accumulation_steps, wodurch auch bei begrenztem GPU-Speicher eine stabile Optimierung ermöglicht wurde. Evaluierungen und Checkpoint-Speicherungen erfolgten schrittbasiert, angepasst an die jeweilige Datensatzgröße und Iterationsgeschwindigkeit.

== Wahl der Lernrate

Für den visuellen Trainingsabschnitt wurde eine Lernrate von 2 × 10⁻⁴ verwendet. Für den nachgelagerten Strategy-Trainingsschritt (Step 2) wurde die Lernrate auf 1 × 10⁻⁵ reduziert, um feinere Anpassungen auf höherer Abstraktionsebene zu ermöglichen.

Vorversuche mit einer durchgängig niedrigen Lernrate im Bereich von 10⁻⁵ zeigten jedoch, dass zu Beginn kaum messbare Fortschritte erzielt wurden. Insbesondere in Kombination mit niedrigen LoRA-Rank, wie sie in vielen bestehenden Arbeiten empfohlen werden, erwies sich die Anpassung als zu restriktiv. Diese empirischen Beobachtungen motivierten die Wahl einer höheren initialen Lernrate in den frühen Trainingsphasen.


== LoRA-Konfiguration und Rank-Wahl

Die LoRA-Adapter wurden mit einem Rank r = 32 konfiguriert. Dieser Wert liegt deutlich über den häufig in der Literatur genannten Bereichen (r = 8 oder r = 16), erwies sich jedoch in der Praxis als notwendig, um eine ausreichende Modellkapazität für die Zielaufgaben bereitzustellen.

Insbesondere bei multimodalen und strategischen Aufgaben zeigte sich, dass niedrigere Ränge die Anpassungsfähigkeit des Modells stark einschränkten. Erst mit r = 32 konnten konsistente Leistungsgewinne beobachtet werden. Die Skalierung der LoRA-Gewichte erfolgte mit lora_alpha = 1.5 × r, was die Stabilität des Trainings weiter verbesserte.

Je nach Trainingsmodus wurden unterschiedliche Zielmodule angepasst:
- Visueller Modus: Projektionen der Selbstaufmerksamkeit (q_proj, k_proj, v_proj, o_proj) sowie der multimodale Projektor.
- Logik-/Strategiemodus: Zusätzlich die Feedforward-Komponenten (gate_proj, up_proj, down_proj), um komplexere transformationsbasierte Anpassungen zu ermöglichen.


== Curriculum Learning und sequentielles Adapter-Merging

Das Training folgte einem Curriculum-Learning-Ansatz, bei dem das Modell schrittweise an zunehmend abstrakte Aufgaben herangeführt wurde. Zunächst lag der Fokus auf visuell dominierten Aufgaben, bevor in späteren Phasen strategische und logisch anspruchsvollere Inhalte trainiert wurden.

Zur technischen Umsetzung dieses Curriculums wurden die LoRA-Adapter sequenziell trainiert und anschließend vollständig in das Hauptmodell gemerged. Nach Abschluss eines Trainingsabschnitts wurden die Adaptergewichte in das Basismodell integriert und der Adapter anschließend entladen. Dieser Ansatz hatte zwei wesentliche Vorteile:
1. Stabilität: Das sequentielle Mergen verhinderte Warnungen und potenzielle Inkonsistenzen beim gleichzeitigen Laden mehrerer Adapter.
2. Explizite Wissensakkumulation: Jede Trainingsphase baute direkt auf den zuvor integrierten Gewichten auf, wodurch das Curriculum explizit im Modellzustand verankert wurde.

Durch dieses Vorgehen entstand eine klare Abfolge von Lernschritten, bei der jede Phase das Ergebnis der vorherigen konsolidierte, anstatt konkurrierende Adapter parallel zu halten.
