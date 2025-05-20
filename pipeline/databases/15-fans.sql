-- 3 first students in the Batch ID=3
-- because Batch 3 is the best!
SELECT origin, sum(fans) as nb_fans
FROM metal_bands
group by
    origin
ORDER BY nb_fans DESC;