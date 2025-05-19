-- 3 first students in the Batch ID=3
-- because Batch 3 is the best!
SELECT tg.name genre, COUNT(tg.name) number_of_shows
FROM
    tv_show_genres tsg
    JOIN tv_genres tg on tg.id = tsg.genre_id
GROUP BY
    tg.name
ORDER BY COUNT(tg.name) DESC;