# Generowanie obrazów twarzy za pomocą sieci GAN

## Autorki projektu
- Alicja Wojciechowska
- Weronika Żygis
## Data wykonania projektu
Maj 2025r.

## Krótki opis projektu (założenia, cele projektowe, funkcje, przewidywane przeznaczenie)
Celem projektu było stworzenie modelu opartego na sieci generatywnej GAN (Generative Adversarial Network), który potrafi generować nowe, realistycznie wyglądające twarze ludzi.  Model bazuje na architekturze DCGAN, która łączy klasyczne założenia GAN-ów z sieciami konwolucyjnymi, co pozwala uzyskać lepsze rezultaty w kontekście generowania obrazów.

Trening przeprowadzono na zbiorze danych CelebFaces Attributes Dataset (CelebA), zawierającym tysiące zdjęć twarzy celebrytów o dużej różnorodności cech wizualnych, takich jak wiek, płeć, fryzura czy wyraz twarzy.

### Założenia:
- Zastosowanie konwolucyjnych i transponowanych warstw konwolucyjnych w dyskryminatorze i generatorze.
- Użycie funkcji aktywacji ReLU (w generatorze) oraz LeakyReLU (w dyskryminatorze) dla lepszej propagacji gradientów.
- Normalizacja wsadowa (batch normalization) w celu stabilizacji procesu uczenia.  

### Funkcje:
- Trening modelu DCGAN na zbiorze danych CelebA.
- Generowanie syntetycznych twarzy o wymiarach 64x64 piksele.
- Wizualizacja wygenerowanych obrazów w trakcie i po zakończeniu treningu.

### Przewidywane przeznaczenie:
Tworzenie realistycznych wizerunków osób, które nie istnieją w rzeczywistości — np. do zastosowań w grach komputerowych, filmach, reklamach, sztuce cyfrowej czy testowaniu systemów rozpoznawania twarzy.

## Wybór technologii informatycznych
- PyTorch 
- NumPy
- Kaggle Api
- Google Colab - przyspieszenie trenowania dzięki wsparciu GPU.

## Projekt architektury aplikacji z uzasadnieniem wyboru technologii

## Projekt bazy danych (użyte tabele w przypadku relacyjnych baz danych)
Nie dotyczy – projekt nie korzysta z klasycznej relacyjnej bazy danych. Dane wejściowe (obrazy) są przechowywane jako pliki w systemie plików.

  
