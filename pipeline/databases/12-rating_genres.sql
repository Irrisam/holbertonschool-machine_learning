-- 3 first students in the Batch ID=3
-- because Batch 3 is the best!
SELECT tg.name, SUM(rate) rating
FROM
    tv_show_genres tsg
    JOIN tv_show_ratings tsr ON tsr.show_id = tsg.show_id
    JOIN tv_genres tg on tg.id = tsg.genre_id
GROUP BY
    tg.name
ORDER BY SUM(rate) DESC;