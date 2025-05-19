-- 3 first students in the Batch ID=3
-- because Batch 3 is the best!
SELECT ts.title, SUM(rate) rating
FROM
    tv_show_ratings tsr
    JOIN tv_shows ts on ts.id = tsr.show_id
GROUP BY
    ts.title
ORDER BY SUM(rate) DESC;