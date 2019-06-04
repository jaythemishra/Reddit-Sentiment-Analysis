CREATE SCHEMA test
CREATE TABLE test.data
(
  id character varying(50),
  created_utc character varying(50),
  linkid character varying(50),
  author_flair_text character varying(50),
  title character varying(250),
  body character varying(10000),
  is_positive character(20) not null,
  is_negative character(20) not null
)

\copy test.data FROM '/home/cs143/project2/sample_data.csv' DELIMITER ',' CSV

SELECT 
  t1.linkid,
  COUNT(t1.id) AS total_comments,
  t2.pos_num / COUNT(t1.id) * 100 AS perc_pos,
  t2.neg_num / COUNT(t1.id) * 100 AS perc_neg
FROM test.data t1
LEFT JOIN (
  SELECT linkid, 
    sum(case is_positive when '1' then 1 else 0 end) AS pos_num,
    sum(case is_negative when '1' then 1 else 0 end) AS neg_num
  FROM test.data
  GROUP BY linkid
) t2 
ON t2.linkid = t1.linkid
GROUP BY t1.linkid, t2.pos_num, t2.neg_num

SELECT linkid, 
COUNT(id) AS total_comments
-- (((sum(case is_positive when '1' then 1 else 0 end)) / count(id))) AS pos_num,
(((sum(case is_negative when '1' then 1 else 0 end)) / count(id))) AS neg_num
FROM test.data
GROUP BY linkid