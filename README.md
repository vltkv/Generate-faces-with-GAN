# Generowanie obrazów twarzy za pomocą sieci GAN

## Autorki projektu
- Alicja Wojciechowska
- Weronika Żygis
## Data wykonania projektu
2025r.

## Opis projektu
Celem projektu było stworzenie modelu opartego na sieci generatywnej GAN (Generative Adversarial Network), który potrafi generować nowe, realistycznie wyglądające twarze ludzi.  Model bazuje na architekturze DCGAN, która łączy klasyczne założenia GAN-ów z sieciami konwolucyjnymi, co pozwala uzyskać lepsze rezultaty w kontekście generowania obrazów.

Trening przeprowadzono na zbiorze danych CelebFaces Attributes Dataset (CelebA), zawierającym tysiące zdjęć twarzy celebrytów o dużej różnorodności cech wizualnych, takich jak wiek, płeć, fryzura czy wyraz twarzy.

### Założenia:
- Zastosowanie konwolucyjnych i transponowanych warstw konwolucyjnych w dyskryminatorze i generatorze.
- Użycie funkcji aktywacji ReLU (w generatorze) oraz LeakyReLU (w dyskryminatorze) dla lepszej propagacji gradientów.
- Normalizacja wsadowa (batch normalization) w celu stabilizacji procesu uczenia.

### Cele projektowe:

### Funkcje:
- Trening modelu DCGAN na zbiorze danych CelebA.
- Generowanie syntetycznych twarzy o wymiarach 64x64 piksele.
- Wizualizacja wygenerowanych obrazów w trakcie i po zakończeniu treningu.
- Możliwość uruchomienia modelu na kilku kartach graficznych (GPU).

### Przewidywane przeznaczenie:
Tworzenie realistycznych wizerunków osób, które nie istnieją w rzeczywistości — np. do zastosowań w grach komputerowych, filmach, reklamach, sztuce cyfrowej czy testowaniu systemów rozpoznawania twarzy.

# Analiza systemowa

## Przypadki użycia
## Diagram przypadków użycia – diagram sekwencji
## Podział prac w projekcie
## Etapy wykonania projektu

# Projekt architektury

## Wybór technologii informatycznych
- PyTorch 
- NumPy
- Kaggle Api
- Google Colab - przyspieszenie trenowania dzięki wsparciu GPU.

## Projekt architektury aplikacji
Projekt składa się z dwóch odrębnych modeli: generatora i dyskryminatora. Zadaniem generatora jest tworzenie fałszywych obrazów, które wyglądają jak obrazy z danych treningowych. Z kolei zadaniem dyskryminatora jest ocena obrazu i określenie, czy pochodzi on z prawdziwego zbioru treningowego, czy został wygenerowany przez generator.

Podczas treningu generator nieustannie stara się przechytrzyć dyskryminatora, generując coraz bardziej realistyczne obrazy. Jednocześnie dyskryminator doskonali się w rozróżnianiu prawdziwych i fałszywych obrazów. Proces ten ma charakter rywalizacji. Równowaga tej „gry” zostaje osiągnięta wtedy, gdy generator tworzy tak realistyczne obrazy, że dyskryminator nie potrafi już ich odróżnić od prawdziwych — i zmuszony jest zgadywać z 50% pewnością, czy dany obraz jest prawdziwy, czy fałszywy.

W projekcie wykorzystujemy DCGAN:

**Dyskryminator** składa się z warstw konwolucyjnych z krokiem (ang. strided convolutions), warstw normalizacji wsadowej (batch normalization) oraz aktywacji LeakyReLU. Jako wejście przyjmuje obraz o rozmiarze 3x64x64 (RGB), a jego wyjściem jest skalarna wartość reprezentująca prawdopodobieństwo, że dany obraz pochodzi z rzeczywistego zbioru danych.

**Generator** z kolei zbudowany jest z warstw transponowanych konwolucyjnych (conv-transpose), warstw normalizacji wsadowej oraz aktywacji ReLU. Jako dane wejściowe przyjmuje wektor latentny z, wylosowany z rozkładu normalnego, a na wyjściu generuje obraz RGB o rozmiarze 3x64x64. Warstwy transponowane umożliwiają przekształcenie wektora latentnego w trójwymiarową strukturę odpowiadającą obrazowi.



## Projekt bazy danych (użyte tabele w przypadku relacyjnych baz danych)
Nie dotyczy – projekt nie korzysta z klasycznej relacyjnej bazy danych. Dane wejściowe (obrazy) są przechowywane jako pliki w systemie plików.

## Przypadki użycia w modułach
## Projekt API backendu
## Diagram klas

# Implementacja i testowanie aplikacji z użyciem wybranego narzędzia do projektowania zespołowego


  
