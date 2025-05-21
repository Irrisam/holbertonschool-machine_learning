-- 3 first students in the Batch ID=3
-- because Batch 3 is the best!
CREATE TABLE IF NOT EXISTS users (
    id int NOT NULL AUTO_INCREMENT,
    email varchar(256),
    name varchar(256),
    UNIQUE (email),
    PRIMARY KEY (id)
);