-- 3 first students in the Batch ID=3
-- because Batch 3 is the best!
SELECT ts.title, tsg.genre_id
FROM
    tv_show_genres tsg
    RIGHT JOIN tv_shows ts on ts.id = tsg.show_id
WHERE
    tsg.genre_id is NULL
ORDER BY ts.title, tsg.genre_id ASC;