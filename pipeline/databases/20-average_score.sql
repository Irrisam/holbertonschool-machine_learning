-- 3 first students in the Batch ID=3
-- because Batch 3 is the best!DELIMITER $$
DELIMITER $$

CREATE PROCEDURE ComputeAverageScoreForUser(
    IN p_user_id INT
)
BEGIN
    DECLARE avg_score FLOAT;
    
    SELECT AVG(score) INTO avg_score
    FROM corrections 
    WHERE user_id = p_user_id;
    
    UPDATE users 
    SET average_score = avg_score
    WHERE id = p_user_id;
END$$

DELIMITER;