# Documentation from the migration of the dialogue acts

### Dialogue acts which don't exist in the gazebo NLI

- confirm_availability
- confirm_plan
- db_query
- inform_action
- inform_activate_emergency_shutdown
- inform_battery_loss
- inform_damage_inspection
- inform_damage_inspection_robot
- inform_deactivate_emergency_shutdown
- inform_emergency_action
- inform_emergency_robot
- inform_emergency_solved
- inform_emergency_status
- inform_inspection
- inform_lost_connection
- inform_mission_completed
- inform_moving
- inform_no_risk_spreading
- inform_plan_selected
- inform_risk
- inform_risk_spreading
- inform_robot_capabilities
- inform_symptom
- inform_time_left
- inform_wind
- make_pa_announcement
- mission_timeout
- request_pa_announcement (same as make_pa_announcement)
- request_plan_responsible
- request_robot_emergency
- request_robot_inspect_damage
- request_robot_type
- start

### States Imported from Gazebo NLI
- info_not_available
- inform_abort_mission
- inform_error
- inform_photo_taken
- inform_returning_to_base
- inform_robot_crash
- inform_robot_progress
- inform_robot_status
- inform_robot_velocity
- request_grounding_location
- request_grounding_robot

### States Modified and Merged with Gazebo NLI
- inform_alert -> inform_alert_emergency
- inform_robot -> inform_robot_available
- inform_battery_level -> inform_robot_battery
- inform_eta -> inform_robot_eta
- inform_robot_status -> inform_robot_location
- intro_stata -> intro_hello

### Unchanged states
- inform_arrival