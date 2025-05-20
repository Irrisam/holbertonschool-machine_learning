-- 3 first students in the Batch ID=3
-- because Batch 3 is the best!
-- Create users table if it doesn't exist
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTO_INCREMENT NOT NULL,
    email VARCHAR(255) NOT NULL,
    name VARCHAR(255),
    country ENUM('US', 'CO', 'TN') NOT NULL DEFAULT 'US',
    UNIQUE (email)
);