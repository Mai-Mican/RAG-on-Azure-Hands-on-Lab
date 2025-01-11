using Microsoft.AspNetCore.Mvc;

namespace Simple.Controllers
{
    /// <summary>
    /// ヘルスチェックリクエストを処理するコントローラー。
    /// </summary>
    [ApiController]
    [Route("[controller]")]
    public class HealthController : ControllerBase
    {
        /// <summary>
        /// アプリケーションのヘルスステータスを取得します。
        /// </summary>
        /// <returns>ヘルスステータスを含む <see cref="IActionResult"/>。</returns>
        [HttpGet]
        public IActionResult GetHealth()
        {
            return Ok(new { status = "ok" });
        }
    }
}
