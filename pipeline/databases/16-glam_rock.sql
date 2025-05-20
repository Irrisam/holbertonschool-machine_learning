-- 3 first students in the Batch ID=3
-- because Batch 3 is the best!
SELECT
    band_name,
    CASE
        WHEN split is null then 2020 - formed
        WHEN split is not null then split - formed
    END lifespan
FROM metal_bands
WHERE
    style LIKE '%Glam rock%'
ORDER BY lifespan DESC;