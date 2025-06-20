# Generowanie obrazów twarzy za pomocą sieci GAN

## Autorki projektu
- Alicja Wojciechowska
- Weronika Żygis
## Data wykonania projektu
2025r.

## Opis projektu

### Cele projektowe:
Celem projektu jest stworzenie modelu opartego na sieci generatywnej GAN (Generative Adversarial Network), którego zadaniem jest generowanie syntetycznych obrazów twarzy o wysokiej jakości.

Główne cele:
- Implementacja modelu bazując na architekturze DCGAN (Deep Convolutional GAN), która łączy klasyczne założenia GAN-ów z sieciami konwolucyjnymi, oraz WGAN (Wasserstein GAN), która poprawia stabilność trenowania poprzez zmianę sposobu w jaki mierzona jest odległość między rozkładami danych rzeczywistych i generowanych.
- Trening modelu na realistycznych danych. W tym celu wykorzystano zbiór danych CelebFaces Attributes Dataset (CelebA), zawierający tysiące zdjęć twarzy celebrytów o dużej różnorodności cech wizualnych, takich jak wiek, płeć, fryzura czy wyraz twarzy.
- Ewaluacja jakości syntetycznych obrazów przy pomocy metryk FID i IS.
- Zapisywanie wyników i modeli do ponownego użycia. Generowanie przykładowych twarzy oraz tworzenie raportów z treningu.

### Założenia:
- Trening modelu DCGAN lub WGAN na zbiorze danych CelebA.
- Implementacja architektury GAN z możliwością wyboru parametrów treningowych (liczba epok, batch size, learing rate itp.).
- Umożliwienie uruchomienia modelu na kilku procesorach graficznych (GPU).
- Realizacja funkcjonalności:
    - Ładowanie i przetwarzanie danych
    - Trening modelu i zapisywanie checkpointów
    - Wizualizacja strat i generowanych obrazów
    - Generowanie syntetycznych obrazów
    - Ocena jakości obrazu za pomocą odpowiednich metryk

### Przewidywane przeznaczenie:
Tworzenie realistycznych anonimowych wizerunków twarzy bez naruszania prywatności — np. do zastosowań w grach komputerowych, filmach, reklamach, sztuce cyfrowej czy testowaniu systemów rozpoznawania twarzy.

## Projekt architektury aplikacji
Projekt składa się z dwóch odrębnych modeli: generatora i dyskryminatora. Zadaniem generatora jest tworzenie fałszywych obrazów, które wyglądają jak obrazy z danych treningowych. Z kolei zadaniem dyskryminatora jest ocena obrazu i określenie, czy pochodzi on z prawdziwego zbioru treningowego, czy został wygenerowany przez generator.

Podczas treningu generator nieustannie stara się przechytrzyć dyskryminatora, generując coraz bardziej realistyczne obrazy. Jednocześnie dyskryminator doskonali się w rozróżnianiu prawdziwych i fałszywych obrazów. Proces ten ma charakter rywalizacji. Równowaga tej „gry” zostaje osiągnięta wtedy, gdy generator tworzy tak realistyczne obrazy, że dyskryminator nie potrafi już ich odróżnić od prawdziwych — i zmuszony jest zgadywać z 50% pewnością, czy dany obraz jest prawdziwy, czy fałszywy.

W projekcie wykorzystujemy DCGAN:

**Dyskryminator** składa się z warstw konwolucyjnych z krokiem (ang. strided convolutions), warstw normalizacji wsadowej (batch normalization) oraz aktywacji LeakyReLU. Jako wejście przyjmuje obraz o rozmiarze 3x64x64 (RGB), a jego wyjściem jest skalarna wartość reprezentująca prawdopodobieństwo, że dany obraz pochodzi z rzeczywistego zbioru danych.

**Generator** z kolei zbudowany jest z warstw transponowanych konwolucyjnych (conv-transpose), warstw normalizacji wsadowej oraz aktywacji ReLU. Jako dane wejściowe przyjmuje wektor latentny z, wylosowany z rozkładu normalnego, a na wyjściu generuje obraz RGB o rozmiarze 3x64x64. Warstwy transponowane umożliwiają przekształcenie wektora latentnego w trójwymiarową strukturę odpowiadającą obrazowi.

# Analiza systemowa
## Uruchamianie projektu
Dokładną dokumentację uruchamiania można zobaczyć [tutaj](how-to-run.md).
## Diagram czynności
![Diagram czynności](diagrams/activity_diagram.png)
## Diagram sekwencji
![Diagram sekwencji](diagrams/sequence_diagram.png)
## Podział prac i etapy wykonania projektu
| Etap | Data       | Nazwa elementu projektu                                                              | Odpowiedzialna                       |
|------|------------|--------------------------------------------------------------------------------------|--------------------------------------|
| I    | 05.03.2025 | Research na temat GANów, wybór podejścia (DCGAN)                                     | Alicja Wojciechowska, Weronika Żygis |
| II   | 10.04.2025 | Wybór zbioru danych i preprocessing                                                  | Alicja Wojciechowska                 |
| III  | 20.04.2025 | Szkic architektury sieci                                                             | Weronika Żygis                       |
| IV   | 05.05.2025 | Implementacja generatora i dyskryminatora.                                           | Alicja Wojciechowska                 |
|      |            | Konfiguracja treningu modelu                                                         | Weronika Żygis                       |
| V    | 08.05.2025 | Wybór funkcji strat dla generatora i dyskryminatora (loss functions)                 | Weronika Żygis                       |
|      |            | Wybór optymalizatora (ADAM)                                                          | Alicja Wojciechowska                 |
| VI   | 10.05.2025 | Uczenie modelu. Śledzenie metryk (porównywanie loss dla generatora i dyskryminatora) | Weronika Żygis                       |
| VII  | 13.05.2025 | Wizualizacja wyników. Wprowadzenie augmentacji i dodanie nowego podejścia – WGAN     | Weronika Żygis                       |
| VIII | 16.05.2025 | Ewaluacja i analiza wyników                                                          | Alicja Wojciechowska                 |
| IX   | 19.05.2025 | Dokumentacja projektu                                                                | Alicja Wojciechowska, Weronika Żygis |

## Użycie narzędzi do programowania zespołowego
W ramach zespołowego podejścia do projektowania i wdrażania modelu generatywnego (DCGAN) wykorzystano narzędzia wspierające współpracę programistyczną, takie jak Git i GitHub, co umożliwiło śledzenie zmian, przypisywanie zadań oraz wspólne rozwijanie kodu. Dodatkowo, w celu skutecznego zarządzania pracą zespołu, zastosowano metodę Kanban, która umożliwiła przejrzyste planowanie, monitorowanie postępów oraz elastyczne reagowanie na napotykane trudności.

# Projekt architektury

- Zastosowanie konwolucyjnych i transponowanych warstw konwolucyjnych w dyskryminatorze i generatorze.
- Użycie funkcji aktywacji ReLU (w generatorze) oraz LeakyReLU (w dyskryminatorze) dla lepszej propagacji gradientów.
- Normalizacja wsadowa (batch normalization) w celu stabilizacji procesu uczenia.

## Wybór technologii informatycznych
- **PyTorch** - biblioteka wykorzystywana do tworzenia i trenowania sieci neuronowych (zarówno GAN i jej odmiany). Obsługuje zarówno CPU, jak i GPU (CUDA), co pozwala na przyspieszenie procesu uczenia. Umożliwia łatwe budowanie modeli, trenowanie ich i manipulowanie gradientami.
- **Torchmetrics** - biblioteka służąca do obliczania metryk jakości generowanych obrazów (FID, IS, MS-SSIM). Ułatwia walidację modeli podczas i po zakończeniu treningu.
- **Torchvision** - rozszerzenie PyTorch zawierające narzędzia do wczytywania i transformacji obrazów, pomocniczne funckje do wizualizacji wyników (np. make_grid).
- **NumPy** - główna bibliotek do obliczeń numerycnzych w Pythonie. Pozwala na operacje na macierzach, konwersję danych i przygotowywanie statystyk pomocnicznych.
- **Matplotlib** - narzędzie do wizualizacji danych. Rysowanie wykresów strat (loss) generatora i dyskryminatora w czasie. Podgląd jakości generowanych obrazów w różnych etapach treningów.
- **KaggleHub** - pozwala na łatwe pobieranie modeli i danych z repozytoriów Kaggle.

## Projekt bazy danych (użyte tabele w przypadku relacyjnych baz danych)
Nie dotyczy – projekt nie korzysta z klasycznej relacyjnej bazy danych. Dane wejściowe (obrazy) są przechowywane jako pliki w systemie plików.

## Diagram komponentów
![Diagram komponentów](diagrams/component_diagram.png)
## Diagram klas
![Diagram klas](diagrams/class_diagram.png)

# Ewaluacja wyników
## Metryki
Po ukończeniu implementacji przystąpiono do testowania i oceny jakości generowanych obrazów. W tym celu zastosowano trzy metryki:

1. FID (Fréchet Inception Distance)
FID mierzy odległość statystyczną między cechami obrazów rzeczywistych a obrazów generowanych, wyekstrahowanymi przez sieć Inception v3. Z przedostatniej warstwy tej sieci pobierane są wektory cech reprezentujące wysokopoziomowe właściwości obrazów (np. układ oczu, kształt twarzy, obecność okularów)
Im niższy wynik FID, tym lepsza jakość oraz większe podobieństwo generowanych obrazów do prawdziwych danych.

    Zalety: 
    - Uwzględnia zarówno jakość obrazu, jak i podobieństwo do rzeczywistych danych.
    - Dobrze koreluje z oceną wizualną człowieka

    Wady:
    - Zakłada rozkład normalny cech.
    - Nie rozróżnia artefaktów lokalnych, bo bazuje na globalnych statystykach. Może nie wychwycić drobnych defektów jak np. dziwnie wyglądające oczy lub tło.

    Interpretacja: 
    - FID bliski 0 oznacza, że obrazy syntetyczne są niemal nie do odróżnienia od rzeczywistych.
    - wysoki FID sugeruje, że model generuje obrazy słabej jakości lub z dużymi różnicami względem danych rzeczywistych.

2. IS (Inception Score)
IS bazując na sieci Inception v3 sprawdza czy model spełnia dwie główne zasady:
- Generowany obraz jest wyraźny i realistyczny. To znaczy, że klasyfikator (Inception v3) z wysokim prawdopodobieństwem przypisuje obraz do danej klasy (jest pewny tego co widzi).
- Zbiór obrazów jest zróżnicowany. 

    Zalety: 
    - Szybkość. IS wymaga tylko przejścia wygenerowanych obrazów przez sieć.
    - Jeśli GAN generuje obrazy z różnych klas (np. psy, koty, auta) – IS dobrze pokazuje zarówno ich wyrazistość, jak i różnorodność.

    Wady:
    - Może dawać niski wynik, gdy sieć klasyfikująca nie rozpozna obrazów jako różnych klas (np. w naszym przypadku generowania twarzy).
    - Nie zawsze wykrywa „mode collapse” (jeśli model generuje ten sam obraz o różnych wariantach koloru – IS może być nadal wysoki).

    Interpretacja: 
    - Im wyższy wynik IS, tym lepsza jakość i różnorodność obrazów. Niska wartość może świadczyć o powtarzalnych lub niskiej jakości generacjach.

3. MS-SSIM (Multi-Scale Structural Similarity Index)
MS-SSIM mierzy strukturalne podobieństwo między parami obrazów przy wykorzystaniu informacji o jasności, kontraście i strukturze, ale w wielu skalach przestrzennych (rozdzielczościach). W kontekście GAN-ów, MS-SSIM służy do oceny różnorodności generowanych obrazów — im niższa wartość MS-SSIM, tym większa różnorodność próbek, ponieważ oznacza to, że obrazy są mniej do siebie podobne.

    Zalety: 
    - Umożliwia ocenę różnorodności generowanych obrazów niezależnie od porównania z rzeczywistymi danymi.
    - Metryka jest wrażliwa na drobne zmiany strukturalne, co pozwala wykrywać zjawisko tzw. mode collapse (czyli generowania niemal identycznych obrazów przez GAN).
    - Szybka w obliczaniu i nie wymaga zaawansowanych ekstraktorów cech.

    Wady:
    - Metryka ocenia tylko jak bardzo obrazy są różne od siebie - nie sprawdza czy są realistyczne i dobrej jakości.
    - Wynik MS-SSIM zależy od liczby i sposobu losowania par obrazów do porównania. Zbyt mała liczba par może prowadzić do niestabilnych lub niewiarygodnych wyników.

    Interpretacja: 
    - MS-SSIM bliski 0 oznacza, że generowane obrazy są różnorodne, GAN generuje szeroki wachlarz przykładów.
    - MS-SSIM powyżej 0.5 wskazuje na duże podobieństwo między próbkami.

Metryki zostały zastosowane po zakończeniu procesu uczenia modelu, umożliwiając obiektywną ocenę postępów, porównanie wariantów architektury oraz analizę wpływu modyfikacji takich jak zastosowanie Wasserstein GAN czy technik augmentacji danych.

Modele, wyniki i porównania są odpowiednich treningów są widoczne w folderach [gan-results](/results/gan-results/), [wgan-results](/results/wgan-results/).
