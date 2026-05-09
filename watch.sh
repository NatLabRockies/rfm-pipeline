watch -n 5 'echo "=== Stage Summary ===" && \
 (test -f artifacts/validation_300_sample_no_caps/stage_summaries.json && \
  jq . artifacts/validation_300_sample_no_caps/stage_summaries.json || echo "Not yet created") && \
 echo && echo "=== File Count ===" && \
 (find artifacts/validation_300_sample_no_caps -maxdepth 1 -type d | xargs -I {} find {} -maxdepth 1 -type f | wc -l) && \
 echo && echo "=== QA Audit ===" && \
 (test -f artifacts/validation_300_sample_no_caps/qa_audit_summary.json && \
  jq ".qa_status" artifacts/validation_300_sample_no_caps/qa_audit_summary.json || echo "Not yet created")'
