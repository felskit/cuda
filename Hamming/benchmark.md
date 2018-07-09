## Wersja GPU bez preprocesingu reduce i losowymi danymi (0 par)
```
Randomizing of 256 sequences 1048576 bits each starting...
Randomizing finished

Computing on GPU starting...
Computing finished
Elapsed time (GPU): 2.480000 seconds

Found 0 pairs with Hamming distance of 1
```
Powyższy czas uzyskałem uruchamiając starszą wersję projektu.

## Wersja GPU z preprocesingiem reduce i losowymi danymi (0 par)
```
Generating 256 sequences 1048576 bits each starting...
Generating finished

Computing on GPU starting...
Computing finished
Elapsed time (GPU): 0.478000 seconds

Found 0 pairs with Hamming distance of 1
```
Powyższy wynik pokazuje przewagę rozwiązania z preprocesingiem.

## Porównanie wersji GPU i CPU przy liczbie par rzędu n (dokł. n-1)
```
Generating 256 sequences 1048576 bits each starting...
Generating finished

Computing on GPU starting...
Computing finished
Elapsed time (GPU): 0.481000 seconds

Found 255 pairs with Hamming distance of 1

Computing on CPU starting...
Computing finished
Elapsed time (CPU): 0.389000 seconds

Found 255 pairs with Hamming distance of 1
```
Dla małej liczby par w ciągach oba rozwiązania osiągają porównywalne czasy.

## Porównanie wersji GPU i CPU przy liczbie par rzędu n^2 (dokł. n^2/4)
```
Generating 256 sequences 1048576 bits each starting...
Generating finished

Computing on GPU starting...
Computing finished
Elapsed time (GPU): 1.941000 seconds

Found 16384 pairs with Hamming distance of 1

Computing on CPU starting...
Computing finished
Elapsed time (CPU): 7.594000 seconds

Found 16384 pairs with Hamming distance of 1
```
Dla znacznie większej liczby par zdecydowaną przewagę ma wersja GPU.