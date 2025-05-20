-- 3 first students in the Batch ID=3
-- because Batch 3 is the best!
DELIMITER $$

CREATE PROCEDURE AddBonus(
    IN p_user_id INT,
    IN p_project_name VARCHAR(255),
    IN p_score INT
)
BEGIN
    DECLARE project_id INT;
    
    SELECT id INTO project_id FROM projects WHERE name = p_project_name LIMIT 1;
    
    IF project_id IS NULL THEN
        INSERT INTO projects (name) VALUES (p_project_name);
        SET project_id = LAST_INSERT_ID();
    END IF;
    
    INSERT INTO corrections (user_id, project_id, score) 
    VALUES (p_user_id, project_id, p_score);
END$$