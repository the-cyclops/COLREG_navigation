using UnityEngine;
using MLAgents;
using MLAgents.Sensors;
using MLAgents.Actuators;


public class BoatAgent : Agent
{
    public Rigidbody rb;

    [SerializeField] private float maxSpeed = 10f;
    
    [SerializeField] private float moveSpeed = 10f;
    [SerializeField] private float turnSpeed = 2f;

    public override void OnEpisodeBegin()
    {
        //TODO:
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // 1. Local Velocity (3 observations: x, y, z)
        // We use InverseTransformDirection to tell the AI speed relative to the boat
        Vector3 localVelocity = transform.InverseTransformDirection(rb.linearVelocity);
        
        // Normalize velocity: AI works best with values between -1 and 1
        // Assuming max boat speed is 10 m/s
        sensor.AddObservation(localVelocity.x / maxSpeed);
        sensor.AddObservation(localVelocity.z / maxSpeed);

        // 2. Angular Velocity (Rotation speed - 1 observation)
        sensor.AddObservation(rb.angularVelocity.y / 2f); 

        // 3. Current Heading (Compass)
        // We use the Sine and Cosine of the Y-angle so there's no "jump" at 360/0 degrees
        float headingRadians = transform.eulerAngles.y * Mathf.Deg2Rad;
        sensor.AddObservation(Mathf.Sin(headingRadians));
        sensor.AddObservation(Mathf.Cos(headingRadians));
    }

    // This method is called every step during training OR by your keyboard in Heuristic mode
    public override void OnActionReceived(ActionBuffers actions)
    {
        // We use Continuous Actions: 
        // Index 0: Throttle (Forward/Back)
        // Index 1: Steering (Left/Right)
        float moveInput = actions.ContinuousActions[0];
        float turnInput = actions.ContinuousActions[1];

        // Apply Force (Forward/Back)
        rb.AddRelativeForce(Vector3.forward * moveInput * moveSpeed);

        // Apply Torque (Rotation)
        rb.AddRelativeTorque(Vector3.up * turnInput * turnSpeed);
    }

    // This maps your keyboard to the Action Buffers
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActions = actionsOut.ContinuousActions;
        
        // Vertical axis (W/S or Up/Down) -> Throttle
        continuousActions[0] = Input.GetAxis("Vertical"); 
        
        // Horizontal axis (A/D or Left/Right) -> Steering
        continuousActions[1] = Input.GetAxis("Horizontal");
    }

}