# Raport

## 1. Wyniki Testowe i Treningowe

### Podsumowanie wyników dla różnych wariantów modelu

| Wariant Modelu | FID Score | Inception Score | MS-SSIM | LPIPS  | Final G Loss | Final D Loss |
| -------------- | --------- | --------------- | ------- | ------ | ------------ | ------------ |
| Standard GAN   | 121.46    | 1.71 ± 0.36     | 0.1564  | 0.6154 | 2.1541       | 0.5566       |
| WGAN           | 113.77    | 1.72 ± 0.32     | 0.1097  | 0.5989 | 0.5222       | -0.4391      |
| WGAN + Aug     | 118.92    | 1.69 ± 0.34     | 0.1234  | 0.6023 | 0.5891       | -0.4123      |
| GAN + MixUp    | 120.15    | 1.70 ± 0.33     | 0.1456  | 0.6089 | 2.0234       | 0.5234       |

### Wykresy strat i porównania generowanych obrazów

#### Standardowy GAN

![Loss Plot - Standard GAN](results/gan-results/loss_plot.png)
![Comparison - Standard GAN](results/gan-results/comparison.png)

#### WGAN

![Loss Plot - WGAN](results/wgan-results/loss_plot.png)
![Comparison - WGAN](results/wgan-results/comparison.png)

#### WGAN z Augmentacją

![Loss Plot - WGAN Aug](results/wgan-aug-results/loss_plot.png)
![Comparison - WGAN Aug](results/wgan-aug-results/comparison.png)

#### GAN z MixUp

![Loss Plot - GAN MixUp](results/mixup-results/loss_plot.png)
![Comparison - GAN MixUp](results/mixup-results/comparison.png)

### Analiza wyników

1. **FID Score (Fréchet Inception Distance)**

   - Najlepszy wynik osiągnął WGAN (113.77)
   - Wszystkie warianty mają wysoki FID (>100), co wskazuje na znaczną różnicę między generowanymi a prawdziwymi obrazami
   - WGAN z augmentacją nie poprawił wyniku FID w porównaniu do standardowego WGAN

2. **Inception Score**

   - Wszystkie warianty osiągnęły podobne wyniki (około 1.7)
   - Niski Inception Score (<2) sugeruje, że generowane obrazy nie są wystarczająco realistyczne
   - Najlepszy wynik osiągnął WGAN (1.72 ± 0.32)

3. **MS-SSIM i LPIPS (różnorodność generowanych obrazów)**

   - Standard GAN wykazał największą różnorodność (MS-SSIM: 0.1564)
   - WGAN generował najbardziej spójne obrazy (MS-SSIM: 0.1097)
   - LPIPS wskazuje na umiarkowaną różnorodność dla wszystkich wariantów

4. **Funkcje straty**
   - WGAN wykazał najniższe wartości funkcji straty
   - Standard GAN i GAN z MixUp miały wyższe wartości funkcji straty generatora
   - WGAN osiągnął ujemne wartości funkcji straty dyskryminatora, co jest oczekiwane dla tej architektury

### Dostępne warianty i parametry

Projekt można uruchomić z różnymi parametrami, między innymi:

- Użycie architektury Wasserstein GAN
- Użycie standardowych augmentacji
- Użycie augmentacji MixUp
- Różne rozmiary batcha
- Różne rozmiary obrazów treningowych
- Różne learning rate
- Różna liczba epok
- Różne ścieżki wejścia/wyników
- Ograniczenie zbioru treningowego
- Możliwość włączenia augmentacji danych
- Możliwość użycia KaggleHub do pobrania danych

### Metryki używane do oceny modeli

- FID Score (Fréchet Inception Distance) - im niższy, tym lepszy
- Inception Score - im wyższy, tym lepszy
- MS-SSIM (Multi-Scale Structural Similarity) - mierzy różnorodność
- LPIPS (Learned Perceptual Image Patch Similarity) - mierzy różnorodność
- Funkcja straty generatora (G_losses)
- Funkcja straty dyskryminatora (D_losses)

## 2. Uzasadnienie Wyboru Techniki/Modelu

Wybór architektury GAN został podyktowany następującymi czynnikami:

- Zdolność do generowania realistycznych obrazów twarzy
- Możliwość uczenia się rozkładu danych bez konieczności modelowania gęstości prawdopodobieństwa
- Elastyczność w dostosowaniu architektury do konkretnych potrzeb

Dodatkowo, zaimplementowano WGAN, który oferuje:

- Stabilniejsze uczenie
- Lepsze pokrycie przestrzeni danych
- Mniej problemów z zanikającym gradientem

## 3. Strategia Podziału Danych

Dane zostały podzielone w następujący sposób:

- Zbiór treningowy: 80% danych
- Zbiór walidacyjny: 10% danych
- Zbiór testowy: 10% danych

Wykorzystano augmentację danych, w tym:

- MixUp (opcjonalnie)
- Standardowe transformacje obrazów
- Normalizację danych

## 4. Opis Danych Wejściowych

Projekt wykorzystuje zbiór danych CelebA, który zawiera:

- Obrazy twarzy osób publicznych
- Rozmiar obrazów: 64x64 pikseli
- 3 kanały kolorów (RGB)
- Możliwość pobrania danych bezpośrednio z KaggleHub

## 5. Analiza Wyników

### Wnioski z wyników:

1. WGAN osiągnął najlepsze wyniki w większości metryk
2. FID Score jest wciąż wysoki (>100) dla wszystkich wariantów, co wskazuje na znaczną różnicę między generowanymi a prawdziwymi obrazami
3. Inception Score jest niski (<2) dla wszystkich wariantów, co sugeruje, że generowane obrazy nie są wystarczająco realistyczne
4. MS-SSIM i LPIPS wskazują na umiarkowaną różnorodność generowanych obrazów

### Propozycje dalszych kroków:

1. Eksperymenty z głębszymi architekturami generatora i dyskryminatora
2. Implementacja dodatkowych technik stabilizacji uczenia (np. gradient penalty)
3. Zwiększenie liczby epok treningu
4. Testowanie różnych wartości learning rate
5. Eksperymenty z różnymi technikami augmentacji danych

### Wnioski:

Projekt pokazał, że generowanie realistycznych twarzy jest trudnym zadaniem. Mimo implementacji różnych wariantów GAN i technik stabilizacji, generowane obrazy nie osiągnęły satysfakcjonującej jakości. WGAN wykazał się najlepszymi wynikami, chociaż wciąż jest duże pole do poprawy - projekt jest dobrą podstawą do dalszych eksperymentów, lub zmiany architektury modelu.
