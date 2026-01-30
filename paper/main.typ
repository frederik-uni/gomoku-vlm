#import "@preview/charged-ieee:0.1.4": ieee
#set text(lang: "de")
#show: ieee.with(
  title: [Multimodal Artificial Intelligence: Post-Training Evaluation Results],
  abstract: [
    Diese Studienarbeit evaluiert den Trainingseffekt eines LoRA-Fine-Tunings für das Vision-Language-Modell Qwen3-VL-2B-Instruct im Kontext des Brettspiels Gomoku. Untersucht werden getrennt visuelle Fragefoki (Wahrnehmung, Zähl- und Musteraufgaben) sowie strategische Fragestellungen zur Zugwahl über definierte Fragefoki und Fragevariationen hinweg. Zur Reduktion formatbedingter Fehlklassifikationen erfolgt die Auswertung mittels LLM-as-a-Judge (LISA) auf einem fest definierten Testdatensatz. Neben einem reinen Visual Fine-Tuning und einem anschließenden Strategy Fine-Tuning wird abschließend ein Visual-Curriculum über vier Spielphasen-Stufen (Q1*--Q4*) betrachtet, um den Einfluss schrittweiser Progression und Rehearsal auf lokale Zugewinne und Forgetting-Effekte zu beurteilen.
  ],
  authors: (
    (
      name: "Frederik Schwarz",
      department: [Informatik (Bsc.)],
      organization: [Hochschule Hof, Fakultät Informatik],
      location: [Matrikelnummer: 00515923],
      email: "frederik.schwarz@hof-university.de",
    ),
    (
      name: "Eugen Ganscha",
      department: [Informatik (Bsc.)],
      organization: [Hochschule Hof, Fakultät Informatik],
      location: [Matrikelnummer: 00514322],
      email: "eganscha@hof-university.de",
    ),
  ),
  bibliography: bibliography("refs.bib"),
)
#set text(lang: "de")
#include "struc.typ"
#include "train.typ"
#include "eval.typ"
