-- 3 first students in the Batch ID=3
-- because Batch 3 is the best!
SELECT ts.title, tg.id
from
    tv_show_genres tsg
    JOIN tv_shows ts on ts.id = tsg.show_id
    JOIN tv_genres tg on tg.id = tsg.genre_id
ORDER BY ts.title, tg.id ASC;