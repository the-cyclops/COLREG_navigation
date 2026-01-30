using UnityEngine;

[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer), typeof(MeshCollider))]
public class HollowCylinderGenerator : MonoBehaviour
{
    public float radius = 10f;
    public float height = 5f;
    public int segments = 64; // Più alto = cerchio più liscio


    // Questa funzione viene chiamata AUTOMATICAMENTE quando cambi un valore nell'Inspector
    void OnValidate()
    {
        // Rigenera il cilindro subito
        if (radius > 0 && height > 0 && segments >= 3)
        {
            GenerateCylinder();
        }
    }

    void Start()
    {
        GenerateCylinder();
    }

    void GenerateCylinder()
    {
        Mesh mesh = new Mesh();
        mesh.name = "HollowCylinder";

        // Vertici
        Vector3[] vertices = new Vector3[(segments + 1) * 2];
        // UVs (per la texture/shader)
        Vector2[] uvs = new Vector2[vertices.Length];
        
        for (int i = 0; i <= segments; i++)
        {
            float angle = (float)i / segments * Mathf.PI * 2;
            float x = Mathf.Cos(angle) * radius;
            float z = Mathf.Sin(angle) * radius;

            // Vertici in basso e in alto
            vertices[i] = new Vector3(x, 0, z);
            vertices[i + segments + 1] = new Vector3(x, height, z);

            // UV mapping
            float u = (float)i / segments;
            uvs[i] = new Vector2(u, 0);
            uvs[i + segments + 1] = new Vector2(u, 1);
        }

        // Triangoli (Le facce)
        int[] triangles = new int[segments * 6];
        for (int i = 0; i < segments; i++)
        {
            int baseIndex = i * 6;
            int vertBottom = i;
            int vertTop = i + segments + 1;

            // Primo triangolo
            triangles[baseIndex] = vertBottom;
            triangles[baseIndex + 1] = vertTop;
            triangles[baseIndex + 2] = vertBottom + 1;

            // Secondo triangolo
            triangles[baseIndex + 3] = vertTop;
            triangles[baseIndex + 4] = vertTop + 1;
            triangles[baseIndex + 5] = vertBottom + 1;
        }

        mesh.vertices = vertices;
        mesh.uv = uvs;
        mesh.triangles = triangles;
        mesh.RecalculateNormals();

        GetComponent<MeshFilter>().mesh = mesh;
        
        // Aggiorna il Collider per la fisica
        GetComponent<MeshCollider>().sharedMesh = mesh;
    }
}