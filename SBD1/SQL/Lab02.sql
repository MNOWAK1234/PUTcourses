-- 1
SELECT NAZWISKO, SUBSTR(ETAT, 1, 2) || ID_PRAC AS KOD
FROM PRACOWNICY;

-- 2
SELECT NAZWISKO, TRANSLATE(NAZWISKO, 'KLM', 'XXX') AS WOJNA_LITEROM
FROM PRACOWNICY;

-- 3
SELECT NAZWISKO
FROM PRACOWNICY
WHERE SUBSTR(NAZWISKO, 1, LENGTH(NAZWISKO) / 2) LIKE '%L%';

-- 4
SELECT NAZWISKO, ROUND(PLACA_POD * 1.15) AS PODWYZKA
FROM PRACOWNICY;

-- 5
SELECT NAZWISKO, PLACA_POD,
PLACA_POD * 0.2 AS INWESTYCJA,
PLACA_POD * 0.2 * POWER(1 + 0.1, 10) AS KAPITAL,
PLACA_POD * 0.2 * POWER(1 + 0.1, 10) - PLACA_POD * 0.2 AS ZYSK
FROM PRACOWNICY;

-- 6
SELECT NAZWISKO, ZATRUDNIONY,
FLOOR((DATE '2000-01-01' - ZATRUDNIONY) / 365) AS STAZ
FROM PRACOWNICY;

-- 7
SELECT NAZWISKO,
TO_CHAR(ZATRUDNIONY, 'month' || CHR(9) || ', DD YYYY') AS DATA_ZATRUDNIENIA
FROM PRACOWNICY;

-- 8
SELECT TO_CHAR(CURRENT_DATE, 'day') AS DZIS
FROM PRACOWNICY
WHERE ROWNUM = 1;

-- 9
SELECT NAZWA, ADRES,
CASE ADRES
WHEN 'PIOTROWO 3A' THEN 'NOWE MIASTO'
WHEN 'WLODKOWICA 16' THEN 'GRUNWALD'
WHEN 'STRZELECKA 14' THEN 'STARE MIASTO'
WHEN 'MIELZYNSKIEGO 30' THEN 'STARE MIASTO'
END AS DZIELNICA
FROM ZESPOLY;

-- 10
SELECT NAZWISKO, PLACA_POD,
CASE
WHEN PLACA_POD > 480 THEN 'POWYZEJ 480'
WHEN PLACA_POD = 480 THEN 'DOKLADNIE 480'
WHEN PLACA_POD < 480 THEN 'PONIZEJ 480'
END AS PROG
FROM PRACOWNICY;

-- 11
SELECT NAZWISKO, PLACA_POD,
DECODE(SIGN(PLACA_POD - 480), '0', 'DOKLADNIE 480', '-1', 'PONIZEJ 480', '1', 'POWYZEJ 480') AS PROG
FROM PRACOWNICY;
