-- 3 first students in the Batch ID=3
-- because Batch 3 is the best!
SELECT city, AVG(value) as avg_temp
from temperatures
group by
    city
ORDER BY AVG(value) DESC;