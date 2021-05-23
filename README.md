# GPU HashTable - Cosmin-Răzvan VANCEA - 333CA

Pentru rezolvarea temei am ales să implementez un Hash Table cu linear-probing
ca metodă de rezolvare a coliziunilor.

## Implementare

Tabela de dispersie este stocată ca un vector de structuri `kv` (cheie-valoare).

### Operația de instanțiere

La instanțierea unei clase de tip `GpuHashTable`, aloc în memoria GPU tabela de
dispersie de capacitate totală `size` cu toate elementele setate la `KEY_INVALID`,
marcând astfel că toate pozițiile sunt neocupate.

### Operația de inserare/actualizare (`insertBatch`)

Fiecare pereche de tip cheie-valoare este scrisă de câte un thread GPU. Kernel-ul
calculează hash-ul cheii, apoi iterează prin tabela `kv` începând de la poziția
`hash % size`, câutând fie:

- un spațiu liber (și inserează acolo perechea)
- un spațiu unde se află deja cheia (și se face actualizarea valorii)

Verificarea și actualizarea unei anumite poziții din tabela `kv` se face atomic
folosind `atomicCAS`, și anume se compară valoarea cheii de la indexul curent cu
`KEY_INVALID` și se scrie noua cheie doar dacă comparația returnează adevărat,
astfel garantându-se că două thread-uri nu pot scrie în aceeași locație (doar
unul va reuși să rescrie cheia de la poziția liberă).

Un caz special este operația de actualizare: nu este nevoie de scriere atomică
deoarece un singur thread (cel asociat perechii cheie-valoare) va încerca să
scrie la poziția în cauză.

Pentru calcularea corectă a load factor-ului, în kernel incrementez un contor
de fiecare dată când se face o operație de INSERT exclusiv. Nu puteam face
acest lucru direct din host deoarece perechile primite în `insertBatch` pot
să necesite o operație de tip insert sau update, acest tip putând fi determinat
doar în momentul în care se interoghează tabela (deci în kernel).

În cazul în care host-ul determină că tabela nu are destule spații libere
pentru a face operația de insert, se execută `RESHAPE` (detalii mai jos).

### Operația de căutare (`getBatch`)

Fiecare cheie este căutată de câte un thread GPU. Kernel-ul calculează hash-ul
cheii, apoi iterează prin tabela `kv` începând de la poziția
`hash % size`, câutând spațiul unde se află cheia. Dacă găsește cheia, salvează
valoarea asociată.

În cazul în care nu se găsește cheia in `size` iterații (adică după ce am
parcurs toată tabela), inseamnă că aceasta nu există și se returneaza valoarea
`KEY_INVALID` (0) pentru cheia asociată.

### Operația de redimensionare (`reshape`)

Se alocă o tabelă nouă, iar pentru fiecare slot din tabela veche există un thread
GPU care va insera perechea cheie-valoare de la acel slot (dacă există) în noua
tabelă. (Re)inserarea este implementată ca mai sus.

În primele încercări de rezolvare, tabela `kv` era redimensionată doar atunci
când nu mai erau destule spații libere pentru o operație de insert, iar
redimensionarea se făcea dublând capacitatea actuală până când:

```text
    new_capacity >= num_of_items_to_insert + num_of_items_already_in_table
```

În alte cuvinte, dublarea se oprea atunci când aveam destule spații libere
în `kv` pentru a realiza cu succes operația de insert. Deși ideea de rezolvare
era corectă, iar testele treceau (dublând mereu dimensiunea se respecta cerința
ca load factor-ul să fie între 0.5 și 1.0), timpul de execuție era lent deoarece
practic redimensionarea se făcea doar atunci când load factor-ul ajungea la o
valoare apropiată de 1.0.

Cum tabela de dispersie folosește în spate o logică de tip linear-probing,
cu cât load factor-ul este mai mare, cu atât distanța dintre poziția ideală a
cheii în tabelă (cea dată de funcția de hash) și poziția reală la care este
plasată cheia (unde se găsește un spațiu liber) devine din ce în ce mai mare,
iar astfel operațiile de căutare și inserare devin mai costisitoare din punct
de vedere temporal. (on the bright side: memoria folosită este aproximativ O(n))

Pentru rezolvarea acestei probleme, am ales să schimb logica de redimensionare
cu una care îmi permite să forțez load factor-ul să fie mereu într-un anumit
interval stabilit. Am ales acest interval să fie [0.65, 0.80], favorizând
astfel într-o mică măsura timpul în detrimentul spațiului.

## Rezultate

```text
------- Test T1 START   ----------

HASH_BATCH_INSERT   count: 1000000          speed: 83M/sec          loadfactor: 65%
HASH_BATCH_GET      count: 1000000          speed: 85M/sec          loadfactor: 65%
----------------------------------------------
AVG_INSERT: 83 M/sec,   AVG_GET: 85 M/sec,      MIN_SPEED_REQ: 10 M/sec

------- Test T1 END     ----------       [ OK RESULT: +20 pts ]



------- Test T2 START   ----------

HASH_BATCH_INSERT   count: 500000           speed: 78M/sec          loadfactor: 65%
HASH_BATCH_INSERT   count: 500000           speed: 69M/sec          loadfactor: 65%
HASH_BATCH_GET      count: 500000           speed: 89M/sec          loadfactor: 65%
HASH_BATCH_GET      count: 500000           speed: 141M/sec         loadfactor: 65%
----------------------------------------------
AVG_INSERT: 74 M/sec,   AVG_GET: 115 M/sec,     MIN_SPEED_REQ: 20 M/sec

------- Test T2 END     ----------       [ OK RESULT: +20 pts ]



------- Test T3 START   ----------

HASH_BATCH_INSERT   count: 125000           speed: 51M/sec          loadfactor: 65%
HASH_BATCH_INSERT   count: 125000           speed: 56M/sec          loadfactor: 65%
HASH_BATCH_INSERT   count: 125000           speed: 49M/sec          loadfactor: 65%
HASH_BATCH_INSERT   count: 125000           speed: 43M/sec          loadfactor: 65%
HASH_BATCH_INSERT   count: 125000           speed: 37M/sec          loadfactor: 65%
HASH_BATCH_INSERT   count: 125000           speed: 57M/sec          loadfactor: 78%
HASH_BATCH_INSERT   count: 125000           speed: 30M/sec          loadfactor: 65%
HASH_BATCH_INSERT   count: 125000           speed: 58M/sec          loadfactor: 74%
HASH_BATCH_GET      count: 125000           speed: 65M/sec          loadfactor: 74%
HASH_BATCH_GET      count: 125000           speed: 95M/sec          loadfactor: 74%
HASH_BATCH_GET      count: 125000           speed: 96M/sec          loadfactor: 74%
HASH_BATCH_GET      count: 125000           speed: 95M/sec          loadfactor: 74%
HASH_BATCH_GET      count: 125000           speed: 95M/sec          loadfactor: 74%
HASH_BATCH_GET      count: 125000           speed: 96M/sec          loadfactor: 74%
HASH_BATCH_GET      count: 125000           speed: 90M/sec          loadfactor: 74%
HASH_BATCH_GET      count: 125000           speed: 85M/sec          loadfactor: 74%
----------------------------------------------
AVG_INSERT: 48 M/sec,   AVG_GET: 90 M/sec,      MIN_SPEED_REQ: 40 M/sec

------- Test T3 END     ----------       [ OK RESULT: +15 pts ]



------- Test T4 START   ----------

HASH_BATCH_INSERT   count: 2500000          speed: 89M/sec          loadfactor: 65%
HASH_BATCH_INSERT   count: 2500000          speed: 77M/sec          loadfactor: 65%
HASH_BATCH_INSERT   count: 2500000          speed: 65M/sec          loadfactor: 65%
HASH_BATCH_INSERT   count: 2500000          speed: 56M/sec          loadfactor: 65%
HASH_BATCH_GET      count: 2500000          speed: 99M/sec          loadfactor: 65%
HASH_BATCH_GET      count: 2500000          speed: 152M/sec         loadfactor: 65%
HASH_BATCH_GET      count: 2500000          speed: 152M/sec         loadfactor: 65%
HASH_BATCH_GET      count: 2500000          speed: 145M/sec         loadfactor: 65%
----------------------------------------------
AVG_INSERT: 72 M/sec,   AVG_GET: 137 M/sec,     MIN_SPEED_REQ: 50 M/sec

------- Test T4 END     ----------       [ OK RESULT: +15 pts ]



------- Test T5 START   ----------

HASH_BATCH_INSERT   count: 20000000         speed: 81M/sec          loadfactor: 65%
HASH_BATCH_INSERT   count: 20000000         speed: 69M/sec          loadfactor: 65%
HASH_BATCH_GET      count: 20000000         speed: 90M/sec          loadfactor: 65%
HASH_BATCH_GET      count: 20000000         speed: 106M/sec         loadfactor: 65%
----------------------------------------------
AVG_INSERT: 75 M/sec,   AVG_GET: 98 M/sec,      MIN_SPEED_REQ: 50 M/sec

------- Test T5 END     ----------       [ OK RESULT: +15 pts ]

TOTAL gpu_hashtable  85/85
```

Operațiile `GET` se execută mai rapid, ceea ce era de așteptat deoarece:

1. kernel-urile nu conțin operații atomice
2. se fac mai puține alocări de memorie în VRAM (doar una pentru input/output)
3. nu se redimensionează tabela
4. se fac doar citiri

Legat de operațiile `INSERT`, viteza de scriere scade cu cât sunt mai multe
elemente inserate în vector, însă crește după redimensionare. Acest lucru se
poate datora coliziunilor și procesului de rezolvare al acestora, după cum am
menționat și anterior în paragraful despre `RESHAPE`. (TL;DR: din cauza
distanței dintre poziția ideală și cea reală la care este plasată perechea
cheie-valoare)

## Bibliografie

- [Laboratoare CUDA ASC](https://ocw.cs.pub.ro/courses/asc/laboratoare/08)
- [Simple Lock Free Hash Table (ideea de a avea 1 thread GPU per element de inserat/căutat)](https://nosferalatu.com/SimpleGPUHashTable.html)
